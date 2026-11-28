//! Unit tests for [`super::ShardedAerisSystem`], [`super::ShardingConfig`],
//! [`super::RoutingSnapshot`], and boundary/rebalance edge cases.

use super::*;
use aeris_types::{FilterConfig, MAX_LOAD_FACTOR, MAX_QUOTIENT_BITS};
use tempfile::TempDir;

fn base_filter_cfg(q: u8) -> FilterConfig {
    FilterConfig::new(q, 14, 64, 20, 1.0 / 256.0).unwrap()
}

#[test]
fn uniform_bounds_cover_full_u64_universe_for_all_counts() {
    for count in [1usize, 2, 3, 4, 7, 8, 16, 64, 256] {
        let bounds = compute_uniform_bounds(count).unwrap();
        assert_eq!(bounds.len(), count);
        assert_eq!(bounds[0].min_key, 0);
        assert_eq!(bounds.last().unwrap().max_key, u64::MAX);
        for w in bounds.windows(2) {
            assert!(w[0].min_key <= w[0].max_key);
            assert_eq!(w[0].max_key.checked_add(1), Some(w[1].min_key));
        }
    }
    assert!(compute_uniform_bounds(0).is_err());
}

#[test]
fn explicit_split_boundaries_validation() {
    let b = compute_bounds_from_split_keys(&[]).unwrap();
    assert_eq!(b, vec![ShardBounds::full_universe()]);

    let b2 = compute_bounds_from_split_keys(&[0, 100, u64::MAX - 1]).unwrap();
    assert_eq!(
        b2,
        vec![
            ShardBounds::new(0, 0).unwrap(),
            ShardBounds::new(1, 100).unwrap(),
            ShardBounds::new(101, u64::MAX - 1).unwrap(),
            ShardBounds::new(u64::MAX, u64::MAX).unwrap(),
        ]
    );

    assert!(compute_bounds_from_split_keys(&[10, 10]).is_err());
    assert!(compute_bounds_from_split_keys(&[20, 10]).is_err());
    assert!(compute_bounds_from_split_keys(&[u64::MAX]).is_err());
}

#[test]
fn median_split_handles_skewed_hotspot_and_extremes() {
    let full = ShardBounds::full_universe();
    let skewed: Vec<Key> = (1_000_000..1_000_100).collect();
    let mid = compute_median_split_key(full, &skewed);
    assert_eq!(mid, 1_000_049);

    let two_at_top = vec![u64::MAX - 1, u64::MAX];
    assert_eq!(compute_median_split_key(full, &two_at_top), u64::MAX - 1);

    let empty_mid = compute_median_split_key(full, &[]);
    assert_eq!(empty_mid, u64::MAX / 2);
}

#[test]
fn routing_binary_search_exact_on_all_shard_boundaries() {
    let dir = TempDir::new().unwrap();
    let cfg = ShardingConfig::new(base_filter_cfg(8)).with_initial_shard_count(5);
    let sys = ShardedAerisSystem::open(dir.path(), cfg).unwrap();
    sys.verify_routing_invariants().unwrap();

    let bounds = sys.shard_bounds();
    assert_eq!(bounds.len(), 5);
    for (sid, b) in &bounds {
        assert_eq!(sys.route_shard_id(b.min_key), *sid);
        assert_eq!(sys.route_shard_id(b.max_key), *sid);
        let mid = b.min_key + (b.max_key - b.min_key) / 2;
        assert_eq!(sys.route_shard_id(mid), *sid);
    }

    let mut keys = Vec::new();
    for (_, b) in &bounds {
        keys.push(b.min_key);
        keys.push(b.max_key);
    }
    for &k in &keys {
        sys.insert(k).unwrap();
        assert!(sys.query_point(k));
    }

    assert!(sys.query_range(QueryRange::new(0, u64::MAX).unwrap()));
    for w in bounds.windows(2) {
        let span = QueryRange::new(w[0].1.max_key, w[1].1.min_key).unwrap();
        assert!(sys.query_range(span));
    }
    sys.verify_all_invariants().unwrap();
}

#[test]
fn flat_routing_and_striped_query_counts_exact_above_and_below_linear_threshold() {
    for shard_count in [1usize, 8, 16, 17, 26] {
        let dir = TempDir::new().unwrap();
        let cfg = ShardingConfig::new(base_filter_cfg(8))
            .with_initial_shard_count(shard_count)
            .with_max_shards(64);
        let sys = Arc::new(ShardedAerisSystem::open(dir.path(), cfg).unwrap());
        let bounds = sys.shard_bounds();
        assert_eq!(bounds.len(), shard_count);

        for (sid, b) in &bounds {
            assert_eq!(sys.route_shard_id(b.min_key), *sid);
            assert_eq!(sys.route_shard_id(b.max_key), *sid);
            sys.insert(b.min_key).unwrap();
            let handle = sys.shard_handle(*sid).unwrap();
            assert_eq!(handle.key_count(), 1);
        }

        const THREADS: usize = 6;
        const QUERIES_PER_THREAD: usize = 50;
        let probe_key = bounds[0].1.min_key;
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let sys_ref = Arc::clone(&sys);
                s.spawn(move || {
                    for _ in 0..QUERIES_PER_THREAD {
                        assert!(sys_ref.query_point(probe_key));
                    }
                });
            }
        });
        let shard0 = sys.shard_handle(bounds[0].0).unwrap();
        assert_eq!(shard0.query_count(), (THREADS * QUERIES_PER_THREAD) as u64);

        assert!(sys.query_range(QueryRange::new(0, u64::MAX).unwrap()));
        for w in bounds.windows(2) {
            let r = QueryRange::new(w[0].1.max_key, w[1].1.min_key).unwrap();
            assert!(sys.query_range(r));
        }
        sys.verify_all_invariants().unwrap();
    }
}

#[test]
fn direct_bypass_matches_concurrent_queries_across_1_shard_and_8_shard_topologies() {
    for &(shard_count, bypass_flag) in &[(1usize, true), (1, false), (8, true), (8, false)] {
        let dir = TempDir::new().unwrap();
        let cfg = ShardingConfig::new(base_filter_cfg(7))
            .with_initial_shard_count(shard_count)
            .with_single_shard_bypass(bypass_flag);
        assert_eq!(cfg.single_shard_bypass, bypass_flag);
        let mut sys = ShardedAerisSystem::open(dir.path(), cfg).unwrap();

        let bounds = sys.shard_bounds();
        assert_eq!(bounds.len(), shard_count);

        let mut inserted = Vec::new();
        for (shard_idx, &(sid, b)) in bounds.iter().enumerate() {
            let step = if shard_count == 1 { 53u64 } else { 97u64 };
            let count = if shard_count == 1 { 200u64 } else { 40u64 };
            for i in 0..count {
                let k = b.min_key.saturating_add(i * step + (shard_idx as u64));
                if b.contains(k) {
                    let assigned = sys.insert_direct(k).unwrap();
                    assert_eq!(assigned, sid);
                    inserted.push(k);
                }
            }
        }
        for &k in &[0u64, u64::MAX] {
            let _ = sys.insert_direct(k).unwrap();
            inserted.push(k);
        }

        for &k in &inserted {
            assert!(sys.query_point_direct(k));
            assert!(sys.query_point(k));
            let r = QueryRange::new(k, k.saturating_add(15)).unwrap();
            assert!(sys.query_range_direct(r));
            assert!(sys.query_range(r));
        }

        for (_, b) in &bounds {
            for probe in 0..100u64 {
                let k = b.min_key.saturating_add(1_000_000 + probe * 131);
                if b.contains(k) {
                    assert_eq!(sys.query_point_direct(k), sys.query_point(k));
                    let r = QueryRange::new(k, k.saturating_add(63)).unwrap();
                    assert_eq!(sys.query_range_direct(r), sys.query_range(r));
                }
            }
        }
        for w in bounds.windows(2) {
            let cross = QueryRange::new(
                w[0].1.max_key.saturating_sub(32),
                w[1].1.min_key.saturating_add(32),
            )
            .unwrap();
            assert_eq!(sys.query_range_direct(cross), sys.query_range(cross));
        }
        let full = QueryRange::new(0, u64::MAX).unwrap();
        assert_eq!(sys.query_range_direct(full), sys.query_range(full));

        let held_handle = sys.shard_handle(bounds[0].0).unwrap();
        let extra_key = bounds[0].1.min_key.saturating_add(7_777_777);
        if bounds[0].1.contains(extra_key) {
            let sid = sys.insert_direct(extra_key).unwrap();
            assert_eq!(sid, bounds[0].0);
            assert!(sys.query_point_direct(extra_key));
            assert_eq!(
                sys.query_range_direct(QueryRange::new(extra_key, extra_key + 7).unwrap()),
                sys.query_range(QueryRange::new(extra_key, extra_key + 7).unwrap()),
            );
        }
        drop(held_handle);

        sys.verify_all_invariants().unwrap();
    }
}

#[test]
fn sharding_config_validation_rejects_invalid_bounds_and_thresholds() {
    let base = base_filter_cfg(8);
    assert!(ShardingConfig::new(base.clone())
        .with_min_shards(0)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_min_shards(4)
        .with_max_shards(2)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_min_shards(4)
        .with_initial_shard_count(2)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_max_shard_q(7)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_max_shard_q(MAX_QUOTIENT_BITS + 1)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_split_load_threshold(f64::NAN)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_split_load_threshold(MAX_LOAD_FACTOR + 0.05)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_merge_load_threshold(-0.1)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_merge_load_threshold(DEFAULT_SPLIT_LOAD_THRESHOLD)
        .validate()
        .is_err());
    assert!(ShardingConfig::new(base.clone())
        .with_split_key_threshold(Some(1))
        .validate()
        .is_err());

    let mut zero_stripes = ShardingConfig::new(base);
    zero_stripes.max_lock_stripes_per_shard = 0;
    assert!(zero_stripes.validate().is_err());
}

#[test]
fn routing_snapshot_and_split_merge_error_paths() {
    assert!(RoutingSnapshot::new(Vec::new()).is_err());

    let dir = TempDir::new().unwrap();
    let cfg = ShardingConfig::new(base_filter_cfg(8))
        .with_initial_shard_count(2)
        .with_min_shards(2)
        .with_max_shards(2);
    let sys = ShardedAerisSystem::open(dir.path(), cfg).unwrap();

    assert!(matches!(
        sys.shard_handle(999),
        Err(AerisError::ShardNotFound(999))
    ));
    assert!(matches!(
        sys.split_shard(999),
        Err(AerisError::ShardNotFound(999))
    ));
    assert!(matches!(
        sys.split_shard(0),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        sys.merge_adjacent_shards(1, 0),
        Err(AerisError::InvalidConfig(_))
    ));
    assert!(matches!(
        sys.merge_adjacent_shards(0, 1),
        Err(AerisError::InvalidConfig(_))
    ));
}
