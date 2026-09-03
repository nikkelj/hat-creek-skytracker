//! RATE_CONTROL joystick mapping: `axis_to_rate` + the adaptive rate
//! gearbox — faithful port of joystick_controller.axis_to_rate /
//! AdaptiveRateMapper (Phase 6 of the Rust port). All timing runs on a
//! caller-supplied monotonic `now` so tests (and the cross-language A/B)
//! drive it deterministically.

pub const JOY_BASE_CEILING: i32 = 5;
pub const JOY_MAX_CEILING: i32 = 9;
pub const JOY_ENGAGE_DEFLECTION: f64 = 0.95;
pub const JOY_RELEASE_DEFLECTION: f64 = 0.70;
pub const JOY_ENGAGE_DELAY_S: f64 = 0.8;
pub const JOY_STEP_UP_S: f64 = 0.5;
pub const JOY_STEP_DOWN_S: f64 = 0.3;
pub const JOY_IDLE_RESET_S: f64 = 0.5;
pub const JOY_REVERSAL_DROP: i32 = 2;
pub const JOY_STALE_RESET_S: f64 = 1.0;

/// Map a (tared) axis value to a discrete rate in -ceiling..=ceiling
/// (0 = stop; ramps from 1 at the deadband edge to `ceiling` by ~0.9).
pub fn axis_to_rate(axis_value: f64, ceiling: i32) -> i32 {
    let v = axis_value.clamp(-1.0, 1.0);
    if v.abs() < 0.01 {
        return 0;
    }
    let normalized = ((v.abs() - 0.01) / (0.9 - 0.01)).min(1.0);
    let rate = ((normalized * ceiling as f64).ceil() as i32).clamp(1, ceiling);
    if v > 0.0 {
        rate
    } else {
        -rate
    }
}

/// The automatic gearbox over axis_to_rate: full deflection is safe at
/// first touch (base ceiling); pinning the stick winds the ceiling up
/// step by step; backing off / releasing / reversing winds it down.
pub struct AdaptiveRateMapper {
    pub base_ceiling: i32,
    pub max_ceiling: i32,
    pub engage_delay_s: f64,
    pub ceiling: i32,
    next_step_up: Option<f64>,
    next_step_down: Option<f64>,
    idle_since: Option<f64>,
    last_signs: [i32; 2],
    last_update: Option<f64>,
}

impl AdaptiveRateMapper {
    pub fn new(base_ceiling: i32, max_ceiling: i32, engage_delay_s: f64) -> Self {
        let base = base_ceiling.clamp(1, 9);
        AdaptiveRateMapper {
            base_ceiling: base,
            max_ceiling: max_ceiling.clamp(base, 9),
            engage_delay_s,
            ceiling: base,
            next_step_up: None,
            next_step_down: None,
            idle_since: None,
            last_signs: [0, 0],
            last_update: None,
        }
    }

    pub fn reset(&mut self) {
        self.ceiling = self.base_ceiling;
        self.next_step_up = None;
        self.next_step_down = None;
        self.idle_since = None;
        self.last_signs = [0, 0];
    }

    /// Advance one control cycle; returns (az_rate, alt_rate).
    pub fn update(&mut self, az_axis: f64, alt_axis: f64, now: f64) -> (i32, i32) {
        if let Some(last) = self.last_update {
            if now - last > JOY_STALE_RESET_S {
                self.reset();
            }
        }
        self.last_update = Some(now);

        // Hard direction flip == overshoot: shed gears immediately.
        for (idx, val) in [az_axis, alt_axis].into_iter().enumerate() {
            let sign = if val >= 0.5 {
                1
            } else if val <= -0.5 {
                -1
            } else {
                0
            };
            if sign != 0 && self.last_signs[idx] != 0 && sign != self.last_signs[idx] {
                self.ceiling = (self.ceiling - JOY_REVERSAL_DROP).max(self.base_ceiling);
                self.next_step_up = None;
            }
            if sign != 0 {
                self.last_signs[idx] = sign;
            }
        }

        let deflection = az_axis.abs().max(alt_axis.abs()).min(1.0);
        if deflection >= JOY_ENGAGE_DEFLECTION {
            // Pinned: wind up after the engage delay, then one step per
            // interval on absolute deadlines.
            self.idle_since = None;
            self.next_step_down = None;
            if self.next_step_up.is_none() {
                self.next_step_up = Some(now + self.engage_delay_s);
            }
            while self.ceiling < self.max_ceiling && now >= self.next_step_up.unwrap() {
                self.ceiling += 1;
                self.next_step_up = Some(self.next_step_up.unwrap() + JOY_STEP_UP_S);
            }
        } else {
            self.next_step_up = None;
            if deflection < 0.01 {
                // Released to center.
                match self.idle_since {
                    None => self.idle_since = Some(now),
                    Some(t0) => {
                        if now - t0 >= JOY_IDLE_RESET_S {
                            self.reset();
                        }
                    }
                }
            } else {
                self.idle_since = None;
            }
            if deflection < JOY_RELEASE_DEFLECTION {
                // Backed off: wind down toward base, faster than the wind-up.
                if self.next_step_down.is_none() {
                    self.next_step_down = Some(now + JOY_STEP_DOWN_S);
                }
                while self.ceiling > self.base_ceiling && now >= self.next_step_down.unwrap() {
                    self.ceiling -= 1;
                    self.next_step_down = Some(self.next_step_down.unwrap() + JOY_STEP_DOWN_S);
                }
            } else {
                // Hysteresis band: hold the gear.
                self.next_step_down = None;
            }
        }

        (
            axis_to_rate(az_axis, self.ceiling),
            axis_to_rate(alt_axis, self.ceiling),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_mapping_endpoints() {
        assert_eq!(axis_to_rate(0.0, 9), 0);
        assert_eq!(axis_to_rate(1.0, 9), 9);
        assert_eq!(axis_to_rate(-1.0, 9), -9);
        assert_eq!(axis_to_rate(0.02, 9), 1);
        assert_eq!(axis_to_rate(1.0, 5), 5);
    }

    /// Service the mapper at 15 Hz like the control loop (gaps > 1 s
    /// trigger the stale reset by design).
    fn drive(m: &mut AdaptiveRateMapper, ax: f64, t0: f64, t1: f64) -> i32 {
        let mut t = t0;
        let mut rate = 0;
        while t <= t1 {
            rate = m.update(ax, 0.0, t).0;
            t += 1.0 / 15.0;
        }
        rate
    }

    #[test]
    fn gearbox_winds_up_and_resets() {
        let mut m = AdaptiveRateMapper::new(5, 9, 0.8);
        // Pinned from t=0: first step-up at 0.8, then every 0.5 -> 9 by 2.9.
        assert_eq!(drive(&mut m, 1.0, 0.0, 0.7), 5);
        assert_eq!(drive(&mut m, 1.0, 0.75, 0.9), 6);
        assert_eq!(drive(&mut m, 1.0, 0.95, 2.9), 9);
        // Release to center for the idle window -> snap back to base.
        drive(&mut m, 0.0, 2.95, 3.55);
        assert_eq!(drive(&mut m, 1.0, 3.6, 3.65), 5);
    }

    #[test]
    fn reversal_sheds_gears() {
        let mut m = AdaptiveRateMapper::new(5, 9, 0.8);
        drive(&mut m, 1.0, 0.0, 2.9); // ceiling 9
        assert_eq!(m.ceiling, 9);
        m.update(-1.0, 0.0, 2.97); // hard flip
        assert_eq!(m.ceiling, 7);
    }
}
