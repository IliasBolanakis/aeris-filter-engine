//! Unit tests for [`super::ProactiveTrigger`] and [`super::ProactiveTriggerConfig`].

use super::*;

#[test]
fn config_validation_enforces_guardrail_and_anti_thrashing_bounds() {
    let base = ProactiveTriggerConfig::default();
    assert!(base.validate().is_ok());

    let cases = [
        ProactiveTriggerConfig {
            margin: 0.99,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            margin: 3.01,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            margin: f64::NAN,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            hard_expand_alpha: 0.96,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            hard_expand_alpha: 0.0,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            contraction_alpha_threshold: 0.50,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            contraction_sustained_checks: 0,
            ..base.clone()
        },
        ProactiveTriggerConfig {
            max_post_contraction_alpha: base.hard_expand_alpha,
            ..base.clone()
        },
    ];
    for bad in &cases {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn eq3_boundary_fires_proactive_expand_precisely_before_saturation() {
    let mut trigger = ProactiveTrigger::new(ProactiveTriggerConfig {
        margin: 1.5,
        hard_expand_alpha: 0.90,
        ..ProactiveTriggerConfig::default()
    })
    .unwrap();

    // N = 10_000, alpha = 0.70 -> remaining_slots = 10_000 * 0.25 = 2_500.
    // reverse_map_entries = 7_000, bulk_load_speed = 10_000 -> T_rebuild = 0.70s.
    // T_rebuild * margin = 1.05s.
    // At ingestion_rate = 2_000 keys/s -> T_full = 1.25s > 1.05s -> None.
    let safe_input = TriggerInput {
        capacity_slots: 10_000,
        current_q: 14,
        current_alpha: 0.70,
        ingestion_rate_ewma: 2_000.0,
        reverse_map_entries: 7_000,
        bulk_load_speed: 10_000.0,
    };
    let eval_safe = trigger.evaluate(&safe_input);
    assert_eq!(eval_safe.action, TriggerAction::None);
    assert!((eval_safe.t_full_secs - 1.25).abs() < 1e-9);
    assert!((eval_safe.t_rebuild_secs - 0.70).abs() < 1e-9);

    // Increase ingestion_rate so T_full = 2_500 / 2_500 = 1.00s <= 1.05s -> ProactiveExpand!
    let urgent_input = TriggerInput {
        ingestion_rate_ewma: 2_500.0,
        ..safe_input
    };
    let eval_urgent = trigger.evaluate(&urgent_input);
    assert_eq!(eval_urgent.action, TriggerAction::ProactiveExpand);
    assert!((eval_urgent.t_full_secs - 1.0).abs() < 1e-9);
}

#[test]
fn zero_ingestion_returns_infinite_t_full_and_respects_hard_expand_alpha() {
    let mut trigger = ProactiveTrigger::default();

    let idle_low = TriggerInput {
        capacity_slots: 4096,
        current_q: 12,
        current_alpha: 0.85,
        ingestion_rate_ewma: 0.0,
        reverse_map_entries: 3400,
        bulk_load_speed: 500_000.0,
    };
    let eval_idle = trigger.evaluate(&idle_low);
    assert!(eval_idle.t_full_secs.is_infinite());
    assert_eq!(eval_idle.action, TriggerAction::None);

    let idle_hard = TriggerInput {
        current_alpha: 0.90,
        ..idle_low
    };
    let eval_hard = trigger.evaluate(&idle_hard);
    assert!(eval_hard.t_full_secs.is_infinite());
    assert_eq!(eval_hard.action, TriggerAction::ProactiveExpand);
}

#[test]
fn contraction_requires_sustained_low_alpha_and_safe_projected_load() {
    let mut trigger = ProactiveTrigger::new(ProactiveTriggerConfig {
        contraction_alpha_threshold: 0.25,
        contraction_sustained_checks: 4,
        max_post_contraction_alpha: 0.65,
        ..ProactiveTriggerConfig::default()
    })
    .unwrap();

    let low = TriggerInput {
        capacity_slots: 4096,
        current_q: 12,
        current_alpha: 0.20,
        ingestion_rate_ewma: 10.0,
        reverse_map_entries: 800,
        bulk_load_speed: 500_000.0,
    };

    // 3 consecutive checks -> still None
    for expected_streak in 1..=3 {
        let eval = trigger.evaluate(&low);
        assert_eq!(eval.action, TriggerAction::None);
        assert_eq!(eval.low_alpha_streak, expected_streak);
    }

    // An interruption above threshold resets streak to 0.
    let mid = TriggerInput {
        current_alpha: 0.30,
        ..low
    };
    let eval_mid = trigger.evaluate(&mid);
    assert_eq!(eval_mid.action, TriggerAction::None);
    assert_eq!(eval_mid.low_alpha_streak, 0);

    // Now 4 consecutive low checks fire Contract on the 4th.
    for _ in 0..3 {
        assert_eq!(trigger.evaluate(&low).action, TriggerAction::None);
    }
    let fired = trigger.evaluate(&low);
    assert_eq!(fired.action, TriggerAction::Contract);
    assert!((fired.projected_post_contraction_alpha - 0.40).abs() < 1e-9);
    assert_eq!(trigger.low_alpha_streak(), 0);

    // Must never contract at MIN_QUOTIENT_BITS.
    let min_q_input = TriggerInput {
        current_q: MIN_QUOTIENT_BITS,
        ..low
    };
    for _ in 0..6 {
        assert_eq!(trigger.evaluate(&min_q_input).action, TriggerAction::None);
    }

    // Must not contract if projected post-contraction alpha (2 * alpha) exceeds max_post_contraction_alpha.
    let mut strict_trigger = ProactiveTrigger::new(ProactiveTriggerConfig {
        contraction_alpha_threshold: 0.25,
        contraction_sustained_checks: 2,
        max_post_contraction_alpha: 0.35,
        ..ProactiveTriggerConfig::default()
    })
    .unwrap();
    // alpha = 0.20 <= 0.25, but 2 * 0.20 = 0.40 > 0.35 -> blocked!
    for _ in 0..5 {
        assert_eq!(strict_trigger.evaluate(&low).action, TriggerAction::None);
    }
}

#[test]
fn ceiling_q_non_finite_metrics_and_runtime_setters_behave_safely() {
    let mut trigger = ProactiveTrigger::default();
    trigger.set_margin(2.0).unwrap();
    assert!(trigger.set_margin(5.0).is_err());
    trigger.set_hard_expand_alpha(0.92).unwrap();
    assert!(trigger.set_hard_expand_alpha(0.99).is_err());

    // At MAX_QUOTIENT_BITS, expansion condition suppresses ProactiveExpand.
    let maxed = trigger.evaluate(&TriggerInput {
        capacity_slots: 1 << 20,
        current_q: MAX_QUOTIENT_BITS,
        current_alpha: f64::NAN, // falls back to MAX_LOAD_FACTOR
        ingestion_rate_ewma: 10_000.0,
        reverse_map_entries: 500_000,
        bulk_load_speed: f64::NAN, // falls back to MIN_BULK_LOAD_SPEED_FLOOR
    });
    assert_eq!(maxed.action, TriggerAction::None);
    assert_eq!(maxed.t_rebuild_secs, 500_000.0);

    // Build streak then reset explicitly.
    let low = TriggerInput {
        capacity_slots: 4096,
        current_q: 12,
        current_alpha: 0.10,
        ingestion_rate_ewma: 0.0,
        reverse_map_entries: 400,
        bulk_load_speed: 500_000.0,
    };
    let _ = trigger.evaluate(&low);
    assert_eq!(trigger.low_alpha_streak(), 1);
    trigger.reset_contraction_streak();
    assert_eq!(trigger.low_alpha_streak(), 0);
}
