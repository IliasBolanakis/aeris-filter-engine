//! Unit tests for [`super::service::BoundedCache`] and response planners.

use super::planner::{build_normal_memory_response, build_normal_tuning_response};
use super::service::BoundedCache;
use super::REFERENCE_SERVER_MODEL_VERSION;
use crate::proto;

#[test]
fn bounded_cache_evicts_oldest_and_updates_existing() {
    let mut cache = BoundedCache::new(2);
    cache.insert("k1".to_string(), 10);
    cache.insert("k2".to_string(), 20);
    assert_eq!(cache.get("k1"), Some(10));

    cache.insert("k1".to_string(), 11);
    assert_eq!(cache.get("k1"), Some(11));

    // Inserting k3 evicts k1 (oldest in FIFO order) while keeping k2.
    cache.insert("k3".to_string(), 30);
    assert_eq!(cache.get("k1"), None);
    assert_eq!(cache.get("k2"), Some(20));
    assert_eq!(cache.get("k3"), Some(30));
}

#[test]
fn normal_tuning_and_memory_planners_handle_edge_inputs() {
    let bad_req = proto::TuningRequest {
        request_id: "r1".to_string(),
        shard_id: 0,
        epoch: 1,
        timestamp_ms: 0,
        current_q: 14,
        current_r: 15,
        current_m: 5,
        payload_budget_c: 20,
        target_epsilon: f64::NAN,
        load_factor: 0.85,
        ingestion_rate_ewma: f64::NAN,
        query_rate: 0.0,
        observed_fpr: 0.0,
        p95_query_len: -1.0,
        p99_query_len: 1.0,
        reverse_map_entries: 100,
        bulk_load_speed: 1000.0,
        recent_query_len_series: vec![],
        recent_ingest_rate_series: vec![f64::INFINITY, -5.0, 42.0],
    };
    let resp = build_normal_tuning_response(&bad_req);
    assert_eq!(resp.model_version, REFERENCE_SERVER_MODEL_VERSION);
    assert!(resp.trigger_proactive_rebuild);
    assert_eq!(resp.forecast_peak_ingest_rate, 42.0);
    assert!(resp.proposed_r >= 12);

    let empty_mem = build_normal_memory_response(&proto::MemoryAllocationRequest {
        request_id: "m0".to_string(),
        epoch: 1,
        total_memory_budget_bytes: 1_000_000,
        shards: vec![],
    });
    assert!(empty_mem.allocations.is_empty());
    assert_eq!(empty_mem.model_version, REFERENCE_SERVER_MODEL_VERSION);
}
