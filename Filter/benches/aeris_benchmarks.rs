//! Production Criterion Benchmark Suite for the Aeris-Based
//! Optimized Range Filtering System.
//!
//! Covers all architectural planes:
//! 1. In-memory query throughput & single-query latency (`R = 1, 32, 1024`, 100k keys).
//! 2. Zipfian correlated (`alpha = 1.5, D = 0.8`) and uncorrelated (`alpha = 1.5`)
//!    range query workloads matching `Aeris Filters.pdf` Section 5.2.
//! 3. Two-phase adversarial latency-probing stability (10% repeat FP queries over real
//!    `AerisFilter` + `ReferenceKvStore` vs unadapted baseline).
//! 4. Deamortized `2x` InfiniFilter expansion with selective rejuvenation (`q = 6 -> 14`)
//!    vs full-rebuild baseline over real `RocksDbReverseMap`.
//! 5. Deterministic Control Plane `DiscreteCostOptimizer::optimize` (`< 5 us`).
//! 6. Scale Plane `RcuAerisFilter::query_range` and 8-shard
//!    `ShardedAerisSystem::query_range`.

use aeris_control::DiscreteCostOptimizer;
use aeris_core::AerisFilter;
use aeris_scale::{RcuAerisFilter, ShardedAerisSystem, ShardingConfig};
use aeris_storage::{KvStoreConfig, ReferenceKvStore, ReverseMapConfig, RocksDbReverseMap};
use aeris_types::{
    AerisError, CostModelParams, CostOptimizer, FilterConfig, Key, QueryRange, RangeFilter,
    ReverseMapStore, WorkloadStats, MAX_QUOTIENT_BITS,
};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

const KEYSPACE_MASK: u64 = (1u64 << 48) - 1;
const SAMPLE_SIZE: usize = 20;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

/// Deterministic Zipfian inverse-CDF sampler (`P(rank = i) proportional to 1 / (i + 1)^alpha`).
struct BenchZipfian {
    cdf: Vec<f64>,
}

impl BenchZipfian {
    fn new(domain_size: usize, alpha: f64) -> Self {
        let mut cdf = Vec::with_capacity(domain_size);
        let mut sum = 0.0_f64;
        for rank in 1..=domain_size {
            sum += 1.0 / (rank as f64).powf(alpha);
            cdf.push(sum);
        }
        let inv = 1.0 / sum;
        for c in &mut cdf {
            *c *= inv;
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self { cdf }
    }

    #[inline]
    fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let u: f64 = rng.gen();
        self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1)
    }
}

fn generate_uniform_keys(seed: u64, count: usize) -> (Vec<Key>, BTreeSet<Key>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut set = BTreeSet::new();
    while set.len() < count {
        set.insert(rng.gen::<u64>() & KEYSPACE_MASK);
    }
    let vec: Vec<Key> = set.iter().copied().collect();
    (vec, set)
}

#[inline]
fn is_empty(key_set: &BTreeSet<Key>, range: QueryRange) -> bool {
    key_set.range(range.start()..=range.end()).next().is_none()
}

fn generate_uniform_empty_queries(
    seed: u64,
    key_set: &BTreeSet<Key>,
    range_len: u64,
    count: usize,
) -> Vec<QueryRange> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let start = rng.gen_range(0..=max_start);
        let range = QueryRange::with_len(start, range_len).expect("range_len >= 1");
        if is_empty(key_set, range) {
            out.push(range);
        }
    }
    out
}

fn generate_correlated_zipfian_queries(
    seed: u64,
    sorted_keys: &[Key],
    key_set: &BTreeSet<Key>,
    range_len: u64,
    alpha: f64,
    correlation_d: f64,
    count: usize,
) -> Vec<QueryRange> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let domain = sorted_keys.len().min(4_096);
    let zipf = BenchZipfian::new(domain, alpha);
    let stride = (sorted_keys.len() / domain).max(1);
    let exp_bits = ((30.0 * (1.0 - correlation_d)).round() as u32).clamp(1, 30);
    let max_offset = (1u64 << exp_bits).max(2);
    let max_start = KEYSPACE_MASK.saturating_sub(range_len);

    let mut anchors = Vec::with_capacity(domain);
    for rank in 0..domain {
        let idx = (rank.wrapping_mul(stride)) % sorted_keys.len();
        let anchor = sorted_keys[idx];
        let mut selected = None;
        for attempt in 0..16u64 {
            let start = anchor
                .saturating_add(rng.gen_range(1..=max_offset).saturating_add(attempt))
                .min(max_start);
            let cand = QueryRange::with_len(start, range_len).expect("valid range");
            if is_empty(key_set, cand) {
                selected = Some(cand);
                break;
            }
        }
        anchors.push(selected.unwrap_or_else(|| {
            QueryRange::with_len(rng.gen_range(0..=max_start), range_len).expect("valid range")
        }));
    }

    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        if rng.gen::<f64>() < correlation_d {
            out.push(anchors[zipf.sample(&mut rng)]);
        } else {
            let start = rng.gen_range(0..=max_start);
            let cand = QueryRange::with_len(start, range_len).expect("valid range");
            if is_empty(key_set, cand) {
                out.push(cand);
            }
        }
    }
    out
}

fn generate_uncorrelated_zipfian_queries(
    seed: u64,
    key_set: &BTreeSet<Key>,
    range_len: u64,
    alpha: f64,
    pool_size: usize,
    count: usize,
) -> Vec<QueryRange> {
    let pool = generate_uniform_empty_queries(seed ^ 0x1357_2468, key_set, range_len, pool_size);
    let zipf = BenchZipfian::new(pool.len(), alpha);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9ABC_DEF0);
    (0..count).map(|_| pool[zipf.sample(&mut rng)]).collect()
}

fn adapt_with_auto_expand(filter: &mut AerisFilter, range: QueryRange, rm: &dyn ReverseMapStore) {
    let mut attempts = 0u8;
    loop {
        match filter.adapt(range, rm) {
            Ok(_) => return,
            Err(AerisError::FilterFull { .. }) if attempts < MAX_QUOTIENT_BITS => {
                filter.expand(rm).expect("expand during benchmark setup");
                attempts += 1;
            }
            Err(err) => panic!("unexpected adapt error: {err}"),
        }
    }
}

/// Group 1: In-memory query throughput (`4,096` queries/iter) and single-query latency
/// over 100,000 seeded uniform keys bulk-loaded into [`AerisFilter`] (`q = 17, r = 12`).
fn bench_query_throughput_in_memory(c: &mut Criterion) {
    const NUM_KEYS: usize = 100_000;
    const BATCH_LEN: usize = 4_096;

    let temp_dir = TempDir::new().expect("temp dir for in-memory query bench");
    let (keys, key_set) = generate_uniform_keys(0xBE01_0001, NUM_KEYS);

    let cfg_short = FilterConfig::new(17, 12, 32, 17, 1.0 / 256.0).expect("valid short config");
    let rm_short = RocksDbReverseMap::open(
        temp_dir.path().join("rm_short"),
        ReverseMapConfig::for_wal_coupled(),
    )
    .expect("open short RM");
    let filter_short =
        AerisFilter::bulk_load(cfg_short, &keys, &rm_short).expect("bulk_load short filter");

    let cfg_long = FilterConfig::new(17, 12, 1024, 22, 1.0 / 256.0).expect("valid long config");
    let rm_long = RocksDbReverseMap::open(
        temp_dir.path().join("rm_long"),
        ReverseMapConfig::for_wal_coupled(),
    )
    .expect("open long RM");
    let filter_long =
        AerisFilter::bulk_load(cfg_long, &keys, &rm_long).expect("bulk_load long filter");

    let point_queries: Vec<Key> =
        generate_uniform_empty_queries(0xBE01_0002, &key_set, 1, BATCH_LEN)
            .into_iter()
            .map(|r| r.start())
            .collect();
    let short_queries = generate_uniform_empty_queries(0xBE01_0003, &key_set, 32, BATCH_LEN);
    let long_queries = generate_uniform_empty_queries(0xBE01_0004, &key_set, 1024, BATCH_LEN);

    let mut group = c.benchmark_group("query_throughput_in_memory_100k");
    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP_TIME)
        .measurement_time(MEASUREMENT_TIME);

    group.throughput(Throughput::Elements(BATCH_LEN as u64));
    group.bench_function("point_queries_r1_batch4096", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &k in &point_queries {
                if filter_short.query_point(black_box(k)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.bench_function("short_range_r32_batch4096", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &r in &short_queries {
                if filter_short.query_range(black_box(r)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.bench_function("long_range_r1024_batch4096", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &r in &long_queries {
                if filter_long.query_range(black_box(r)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    // Dedicated single-query latency benchmarks (exact Criterion median/mean per single probe).
    group.throughput(Throughput::Elements(1));
    let mut pt_idx = 0usize;
    group.bench_function("single_query_point_r1", |b| {
        b.iter(|| {
            let k = point_queries[pt_idx & (BATCH_LEN - 1)];
            pt_idx = pt_idx.wrapping_add(1);
            black_box(filter_short.query_point(black_box(k)))
        });
    });

    let mut sr_idx = 0usize;
    group.bench_function("single_query_short_range_r32", |b| {
        b.iter(|| {
            let r = short_queries[sr_idx & (BATCH_LEN - 1)];
            sr_idx = sr_idx.wrapping_add(1);
            black_box(filter_short.query_range(black_box(r)))
        });
    });

    let mut lr_idx = 0usize;
    group.bench_function("single_query_long_range_r1024", |b| {
        b.iter(|| {
            let r = long_queries[lr_idx & (BATCH_LEN - 1)];
            lr_idx = lr_idx.wrapping_add(1);
            black_box(filter_long.query_range(black_box(r)))
        });
    });

    group.finish();
}

/// Group 2: Zipfian correlated (`alpha = 1.5, D = 0.8`) and uncorrelated (`alpha = 1.5`)
/// range query batches (`Throughput::Elements(4096)`).
fn bench_zipfian_workloads(c: &mut Criterion) {
    const NUM_KEYS: usize = 100_000;
    const BATCH_LEN: usize = 4_096;

    let temp_dir = TempDir::new().expect("temp dir for zipfian bench");
    let (keys, key_set) = generate_uniform_keys(0xBE02_0001, NUM_KEYS);
    let cfg = FilterConfig::new(17, 12, 32, 17, 1.0 / 256.0).expect("valid zipfian config");
    let rm = RocksDbReverseMap::open(
        temp_dir.path().join("rm_zipf"),
        ReverseMapConfig::for_wal_coupled(),
    )
    .expect("open zipfian RM");
    let filter = AerisFilter::bulk_load(cfg, &keys, &rm).expect("bulk_load zipfian filter");

    let correlated =
        generate_correlated_zipfian_queries(0xBE02_0002, &keys, &key_set, 32, 1.5, 0.8, BATCH_LEN);
    let uncorrelated =
        generate_uncorrelated_zipfian_queries(0xBE02_0003, &key_set, 32, 1.5, 4_096, BATCH_LEN);

    let mut group = c.benchmark_group("zipfian_workloads_100k");
    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP_TIME)
        .measurement_time(MEASUREMENT_TIME)
        .throughput(Throughput::Elements(BATCH_LEN as u64));

    group.bench_function("zipfian_correlated_alpha1p5_d0p8", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &r in &correlated {
                if filter.query_range(black_box(r)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.bench_function("zipfian_uncorrelated_alpha1p5", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &r in &uncorrelated {
                if filter.query_range(black_box(r)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.finish();
}

/// Group 3: Adversarial latency-probing stability (10% repeat FP queries) comparing
/// adapted [`AerisFilter`] + [`ReferenceKvStore`] lookup path vs unadapted baseline.
fn bench_adversarial_stability(c: &mut Criterion) {
    const NUM_KEYS: usize = 10_000;
    const WARMUP_QUERIES: usize = 20_000;
    const ADV_BATCH_LEN: usize = 1_024;
    const REPEAT_COUNT: usize = ADV_BATCH_LEN / 10;

    let temp_dir = TempDir::new().expect("temp dir for adversarial bench");
    let kv_store = ReferenceKvStore::open_with_config(
        temp_dir.path().join("kv_db"),
        KvStoreConfig::for_wal_coupled(),
    )
    .expect("open ReferenceKvStore");
    let rm = RocksDbReverseMap::open(
        temp_dir.path().join("rm"),
        ReverseMapConfig::for_wal_coupled(),
    )
    .expect("open RocksDbReverseMap");

    let (keys, key_set) = generate_uniform_keys(0xBE03_0001, NUM_KEYS);
    for &k in &keys {
        kv_store.put(k, &k.to_be_bytes()).expect("kv put");
    }
    kv_store.flush().expect("kv flush");

    let cfg = FilterConfig::new(14, 10, 32, 15, 1.0 / 64.0).expect("valid adversarial cfg");
    let mut adapted_filter = AerisFilter::bulk_load(cfg, &keys, &rm).expect("bulk_load");
    let unadapted_filter = adapted_filter.clone();

    // Warm-up: discover genuine FPs and adapt the adaptive filter while leaving baseline unadapted.
    let warmup = generate_uniform_empty_queries(0xBE03_0002, &key_set, 32, WARMUP_QUERIES);
    let mut discovered_fps = Vec::new();
    let mut seen_fps = HashSet::new();
    for &q in &warmup {
        if unadapted_filter.query_range(q) && seen_fps.insert(q) {
            discovered_fps.push(q);
        }
        if adapted_filter.query_range(q) {
            adapt_with_auto_expand(&mut adapted_filter, q, &rm);
        }
    }
    assert!(
        !discovered_fps.is_empty(),
        "warm-up must discover at least 1 real FP for adversarial benchmark"
    );

    let fresh =
        generate_uniform_empty_queries(0xBE03_0003, &key_set, 32, ADV_BATCH_LEN - REPEAT_COUNT);
    let mut adv_rng = ChaCha8Rng::seed_from_u64(0xBE03_0004);
    let mut adv_batch = Vec::with_capacity(ADV_BATCH_LEN);
    let mut fresh_idx = 0usize;
    for step in 0..ADV_BATCH_LEN {
        if step.is_multiple_of(10) && (step / 10) < REPEAT_COUNT {
            adv_batch.push(discovered_fps[adv_rng.gen_range(0..discovered_fps.len())]);
        } else if fresh_idx < fresh.len() {
            adv_batch.push(fresh[fresh_idx]);
            fresh_idx += 1;
        }
    }

    let mut group = c.benchmark_group("adversarial_stability_10pct_repeat_fps");
    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP_TIME)
        .measurement_time(MEASUREMENT_TIME)
        .throughput(Throughput::Elements(adv_batch.len() as u64));

    group.bench_function("aeris_adapted_plus_kv_lookup", |b| {
        b.iter(|| {
            let mut disk_checks = 0usize;
            for &q in &adv_batch {
                if adapted_filter.query_range(black_box(q)) {
                    let _ = black_box(kv_store.range_non_empty(q).expect("kv range check"));
                    disk_checks += 1;
                }
            }
            black_box(disk_checks)
        });
    });

    group.bench_function("unadapted_baseline_plus_kv_lookup", |b| {
        b.iter(|| {
            let mut disk_checks = 0usize;
            for &q in &adv_batch {
                if unadapted_filter.query_range(black_box(q)) {
                    let _ = black_box(kv_store.range_non_empty(q).expect("kv range check"));
                    disk_checks += 1;
                }
            }
            black_box(disk_checks)
        });
    });

    group.finish();
}

/// Group 4: 8 consecutive `2x` expansions (`q = 6 -> 14`, real [`RocksDbReverseMap`])
/// comparing deamortized InfiniFilter selective rejuvenation (`expand`) vs full-rebuild
/// baseline (`rebuild_from_keys`).
fn bench_expansion_deamortized_vs_full_rebuild(c: &mut Criterion) {
    const INITIAL_Q: u8 = 6;
    const STEPS: usize = 8;
    const BATCH_SIZES: [usize; STEPS] = [32, 32, 64, 128, 256, 512, 1_024, 2_048];
    let total_keys: usize = BATCH_SIZES.iter().sum();

    // eps = 1/16 -> r_min = 7: triggers real selective rejuvenation on step 7 (8th expansion).
    let initial_cfg =
        FilterConfig::new(INITIAL_Q, 7, 32, 12, 1.0 / 16.0).expect("valid expansion bench cfg");
    let (mut shuffled, _) = generate_uniform_keys(0xBE04_0001, total_keys);
    let mut rng = ChaCha8Rng::seed_from_u64(0xBE04_0002);
    for i in (1..shuffled.len()).rev() {
        let j = rng.gen_range(0..=i);
        shuffled.swap(i, j);
    }

    let mut batches: Vec<Vec<Key>> = Vec::with_capacity(STEPS);
    let mut off = 0usize;
    for &sz in &BATCH_SIZES {
        batches.push(shuffled[off..off + sz].to_vec());
        off += sz;
    }

    let rm_cfg = ReverseMapConfig::for_wal_coupled().with_lz4_compression(false);

    let mut group = c.benchmark_group("expansion_8x_q6_to_q14");
    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP_TIME)
        .measurement_time(MEASUREMENT_TIME)
        .throughput(Throughput::Elements(total_keys as u64));

    group.bench_function("deamortized_selective_rejuvenation", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().expect("temp dir");
                let rm = RocksDbReverseMap::open(dir.path().join("rm"), rm_cfg.clone())
                    .expect("open deamort RM");
                let filter = AerisFilter::new(initial_cfg.clone()).expect("init filter");
                (dir, rm, filter)
            },
            |(_dir, rm, mut filter)| {
                let mut total_rejuv = 0usize;
                for batch in &batches {
                    let mut rm_entries = Vec::with_capacity(batch.len());
                    for &k in batch {
                        let fp = filter.insert(k).expect("insert");
                        rm_entries.push((fp, k));
                    }
                    rm.append_batch(&rm_entries).expect("append_batch");
                    let rep = filter.expand(&rm).expect("expand");
                    total_rejuv += rep.boxes_rejuvenated;
                }
                debug_assert!(total_rejuv > 0);
                black_box((filter.config().q, total_rejuv))
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("full_rebuild_baseline", |b| {
        b.iter_batched(
            || {
                let dir = TempDir::new().expect("temp dir");
                let rm = RocksDbReverseMap::open(dir.path().join("rm_0"), rm_cfg.clone())
                    .expect("open baseline RM 0");
                let filter = AerisFilter::new(initial_cfg.clone()).expect("init filter");
                (dir, rm, filter)
            },
            |(dir, mut rm, mut filter)| {
                for (step, batch) in batches.iter().enumerate() {
                    let mut rm_entries = Vec::with_capacity(batch.len());
                    for &k in batch {
                        let fp = filter.insert(k).expect("insert");
                        rm_entries.push((fp, k));
                    }
                    rm.append_batch(&rm_entries).expect("append_batch");

                    let next_cfg = filter.config().expanded().expect("expanded cfg");
                    let persisted = rm.scan_all_keys().expect("scan_all_keys");
                    let next_rm = RocksDbReverseMap::open(
                        dir.path().join(format!("rm_{}", step + 1)),
                        rm_cfg.clone(),
                    )
                    .expect("open next baseline RM");
                    filter = AerisFilter::rebuild_from_keys(next_cfg, persisted, &next_rm)
                        .expect("rebuild_from_keys");
                    rm = next_rm;
                }
                black_box(filter.config().q)
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// Group 5: Deterministic Control Plane [`DiscreteCostOptimizer::optimize`]
/// (verifying sub-5 microsecond zero-allocation discrete optimization).
fn bench_control_plane_optimizer(c: &mut Criterion) {
    let optimizer = DiscreteCostOptimizer::new();
    let current = FilterConfig::new(16, 15, 32, 20, 1.0 / 256.0).expect("valid opt config");
    let cost_model = CostModelParams::default();

    let scenarios: &[(&str, u64)] = &[
        ("short_span_r16", 16),
        ("medium_span_r256", 256),
        ("wide_bypass_span_1m", 1_000_000),
    ];

    let mut group = c.benchmark_group("control_plane_discrete_optimizer");
    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP_TIME)
        .measurement_time(MEASUREMENT_TIME)
        .throughput(Throughput::Elements(1));

    for &(label, span) in scenarios {
        let stats = WorkloadStats {
            effective_query_span: span,
            p50_query_len: span / 2 + 1,
            p95_query_len: span,
            p99_query_len: span,
            ..WorkloadStats::default()
        };
        group.bench_with_input(BenchmarkId::new("optimize", label), &stats, |b, s| {
            b.iter(|| {
                black_box(optimizer.optimize(
                    black_box(s),
                    black_box(&current),
                    black_box(&cost_model),
                ))
            });
        });
    }

    group.finish();
}

/// Group 6: Scale Plane [`RcuAerisFilter::query_range`] and 8-shard
/// [`ShardedAerisSystem::query_range`] (`Throughput::Elements(1024)`).
fn bench_scale_plane_rcu_and_sharded(c: &mut Criterion) {
    const NUM_KEYS: usize = 50_000;
    const NUM_SHARDS: usize = 8;
    const QUERY_BATCH: usize = 1_024;

    let temp_dir = TempDir::new().expect("temp dir for scale plane bench");
    let (keys, key_set) = generate_uniform_keys(0xBE06_0001, NUM_KEYS);

    // 1. Seed RcuAerisFilter (50,000 keys, q = 16 -> 65,536 slots).
    let rcu_cfg = FilterConfig::new(16, 12, 32, 17, 1.0 / 256.0).expect("valid RCU bench cfg");
    let rcu_rm: Arc<dyn ReverseMapStore> = Arc::new(
        RocksDbReverseMap::open(
            temp_dir.path().join("rcu_rm"),
            ReverseMapConfig::for_wal_coupled(),
        )
        .expect("open RCU RM"),
    );
    let rcu_inner = AerisFilter::bulk_load(rcu_cfg, &keys, rcu_rm.as_ref()).expect("bulk_load RCU");
    let rcu_filter = RcuAerisFilter::new(0, rcu_inner, rcu_rm);

    // 2. Seed 8-shard ShardedAerisSystem (50,000 keys spread uniformly across 8 shards).
    let shard_cfg = FilterConfig::new(14, 12, 32, 17, 1.0 / 256.0).expect("valid shard cfg");
    let sharding_cfg = ShardingConfig::new(shard_cfg)
        .with_initial_shard_count(NUM_SHARDS)
        .with_min_shards(NUM_SHARDS)
        .with_max_shards(NUM_SHARDS)
        .with_reverse_map_config(ReverseMapConfig::for_wal_coupled());
    let sharded = Arc::new(
        ShardedAerisSystem::open(temp_dir.path().join("sharded"), sharding_cfg)
            .expect("open 8-shard ShardedAerisSystem"),
    );

    let bounds = sharded.shard_bounds();
    let per_shard = NUM_KEYS / NUM_SHARDS;
    let mut handles = Vec::with_capacity(NUM_SHARDS);
    for (idx, &(_, b)) in bounds.iter().enumerate() {
        let sys = Arc::clone(&sharded);
        handles.push(thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(0xBE06_1000 + idx as u64);
            let span = b.max_key - b.min_key;
            let mut seen = HashSet::with_capacity(per_shard);
            while seen.len() < per_shard {
                let k = b.min_key + (rng.gen::<u64>() % span);
                if seen.insert(k) {
                    sys.insert(k).expect("bench shard insert");
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("bench seed thread panicked");
    }

    let rcu_queries = generate_uniform_empty_queries(0xBE06_2001, &key_set, 32, QUERY_BATCH);
    let mut sharded_rng = ChaCha8Rng::seed_from_u64(0xBE06_2002);
    let sharded_queries: Vec<QueryRange> = (0..QUERY_BATCH)
        .map(|_| {
            let start = sharded_rng.gen::<u64>().min(u64::MAX - 32);
            QueryRange::with_len(start, 32).expect("valid sharded query")
        })
        .collect();

    let mut group = c.benchmark_group("scale_plane_rcu_and_sharded_50k");
    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP_TIME)
        .measurement_time(MEASUREMENT_TIME)
        .throughput(Throughput::Elements(QUERY_BATCH as u64));

    group.bench_function("rcu_aeris_filter_query_range_batch1024", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &r in &rcu_queries {
                if rcu_filter.query_range(black_box(r)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.bench_function("sharded_8_shards_query_range_batch1024", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for &r in &sharded_queries {
                if sharded.query_range(black_box(r)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_query_throughput_in_memory,
    bench_zipfian_workloads,
    bench_adversarial_stability,
    bench_expansion_deamortized_vs_full_rebuild,
    bench_control_plane_optimizer,
    bench_scale_plane_rcu_and_sharded,
);
criterion_main!(benches);
