pub const RANGE_LIMIT: f64 = 10.0;

pub fn clamp_pair(min_val: f64, max_val: f64, limit: f64) -> (f64, f64) {
    let mut lo = min_val.min(max_val);
    let mut hi = min_val.max(max_val);

    lo = lo.clamp(-limit, limit);
    hi = hi.clamp(-limit, limit);

    if (hi - lo).abs() < 1e-9 {
        let center = (hi + lo) / 2.0;
        let mut new_lo = center - 1.0;
        let mut new_hi = center + 1.0;
        new_lo = new_lo.clamp(-limit, limit);
        new_hi = new_hi.clamp(-limit, limit);
        if (new_hi - new_lo).abs() < 1e-9 {
            new_lo = (-limit).min(new_lo);
            new_hi = (limit).max(new_hi);
        }
        lo = new_lo;
        hi = new_hi;
    }

    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_pair_reorders() {
        let (lo, hi) = clamp_pair(3.0, -2.0, RANGE_LIMIT);
        assert_eq!(lo, -2.0);
        assert_eq!(hi, 3.0);
    }

    #[test]
    fn test_clamp_pair_limits() {
        let (lo, hi) = clamp_pair(-20.0, 5.0, RANGE_LIMIT);
        assert_eq!(lo, -10.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn test_clamp_pair_expands_zero_width() {
        let (lo, hi) = clamp_pair(2.0, 2.0, RANGE_LIMIT);
        assert!(hi > lo);
    }
}
