//! Sorted-multiset maintenance and 40%-breakdown outlier-clipped quantile math.

use super::config::EstimatorConfig;

/// Exact quantile summary over the outlier-clipped sliding window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ClippedQuantiles {
    pub(super) p50: u64,
    pub(super) p95: u64,
    pub(super) p99: u64,
    pub(super) clipped_outliers: u64,
}

impl ClippedQuantiles {
    /// Safe empty-window default quantiles (`p50 = p95 = p99 = 1`, `0` clipped outliers).
    pub(super) const EMPTY: Self = Self {
        p50: 1,
        p95: 1,
        p99: 1,
        clipped_outliers: 0,
    };
}

/// Replaces one occurrence of `evicted_len` with `new_len` inside the non-decreasing
/// sorted multiset `sorted` using at most a single contiguous `copy_within` shift
/// instead of separate `Vec::remove` and `Vec::insert` passes.
///
/// # Preconditions
/// `sorted` MUST be sorted in non-decreasing order and `evicted_len != new_len`.
///
/// # Complexity
/// `O(log N + |new_idx - old_idx|)` time and `O(1)` auxiliary space.
#[inline]
pub(super) fn replace_sorted_sample(sorted: &mut Vec<u64>, evicted_len: u64, new_len: u64) {
    if let Ok(old_idx) = sorted.binary_search(&evicted_len) {
        if new_len > evicted_len {
            let rel = sorted[old_idx + 1..].partition_point(|&v| v < new_len);
            let new_idx = old_idx + rel;
            if new_idx > old_idx {
                sorted.copy_within((old_idx + 1)..=new_idx, old_idx);
            }
            sorted[new_idx] = new_len;
        } else {
            let new_idx = sorted[..old_idx].partition_point(|&v| v < new_len);
            if new_idx < old_idx {
                sorted.copy_within(new_idx..old_idx, new_idx + 1);
            }
            sorted[new_idx] = new_len;
        }
    } else {
        // Defensive fallback (unreachable while ring buffer mirrors `sorted`).
        let insert_idx = sorted.partition_point(|&v| v < new_len);
        sorted.insert(insert_idx, new_len);
    }
}

/// Computes exact quantiles over `sorted` after applying 40%-breakdown `raw_p60`-anchored
/// outlier clipping and `max_clip_span` capping.
///
/// # Preconditions
/// `sorted` MUST be sorted in non-decreasing order.
///
/// # Complexity
/// `O(1)` when no outliers exceed `clip_cap`, otherwise `O(log N)` time and `O(1)` space.
pub(super) fn compute_clipped_quantiles(
    sorted: &[u64],
    config: &EstimatorConfig,
    min_clip_samples: usize,
) -> ClippedQuantiles {
    let n = sorted.len();
    if n == 0 {
        return ClippedQuantiles::EMPTY;
    }

    let max_val = sorted[n - 1];
    let (inliers, clipped_outliers) = if n >= min_clip_samples {
        let raw_p60 = nearest_rank_quantile_ratio(sorted, 3, 5);
        let outlier_cap = raw_p60
            .saturating_mul(config.outlier_median_multiplier)
            .max(config.outlier_floor_span);
        let clip_cap = outlier_cap.min(config.max_clip_span);

        if max_val <= clip_cap {
            // Fast path: all samples lie within both `clip_cap` and `outlier_cap` (zero binary searches).
            (sorted, 0)
        } else {
            // Statistical outliers strictly above `outlier_cap` are excluded from the inlier
            // slice. Because `raw_p60 <= outlier_cap`, `within_outlier_cap >= ceil(0.60 * n) >= 1`.
            let within_outlier_cap = sorted.partition_point(|&v| v <= outlier_cap);
            let inlier_count = within_outlier_cap.max(1);
            let within_clip_cap = if clip_cap == outlier_cap {
                within_outlier_cap
            } else {
                sorted[..within_outlier_cap].partition_point(|&v| v <= clip_cap)
            };
            let clipped = (n - within_clip_cap) as u64;
            (&sorted[..inlier_count], clipped)
        }
    } else if max_val <= config.max_clip_span {
        (sorted, 0)
    } else {
        let within_max = sorted.partition_point(|&v| v <= config.max_clip_span);
        let clipped = (n - within_max) as u64;
        (sorted, clipped)
    };

    let cap = config.max_clip_span;
    ClippedQuantiles {
        p50: exact_median_sorted(inliers).min(cap).max(1),
        p95: nearest_rank_quantile_ratio(inliers, 19, 20).min(cap).max(1),
        p99: nearest_rank_quantile_ratio(inliers, 99, 100)
            .min(cap)
            .max(1),
        clipped_outliers,
    }
}

/// Returns the exact integer median of a non-empty sorted slice without overflow.
///
/// For even lengths `2k`, averages the two middle elements `(a + b) / 2` so exact
/// 50/50 bimodal distributions do not bias arbitrarily toward one side.
#[inline]
pub(super) fn exact_median_sorted(sorted: &[u64]) -> u64 {
    let n = sorted.len();
    debug_assert!(n > 0);
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        let lo = sorted[(n / 2) - 1];
        let hi = sorted[n / 2];
        lo + ((hi - lo) / 2)
    }
}

/// Computes the exact nearest-rank percentile for rational `num / den in (0, 1]`
/// over a non-empty sorted slice using pure integer arithmetic (`div_ceil`),
/// avoiding external `f64::ceil()` libm calls on baseline x86-64 targets.
#[inline]
pub(super) fn nearest_rank_quantile_ratio(sorted: &[u64], num: usize, den: usize) -> u64 {
    let n = sorted.len();
    debug_assert!(n > 0 && num > 0 && num <= den);
    let rank = n.saturating_mul(num).div_ceil(den);
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}
