
include!(concat!(env!("OUT_DIR"), "/log_table.rs"));

/// Number of mantissa bits below the bits that index `LOG_TABLE`; used for interpolation.
const FRACTION_BITS: u32 = 23 - LOG_TABLE_BITS;


/// Approximates the natural logarithm of `x` using a precomputed lookup table.
///
/// `x` is split into its binary exponent `e` and mantissa `m` in `[1, 2)`, so that `ln(x) = e * ln(2) + ln(m)`.
/// `ln(m)` is interpolated linearly between the two nearest entries of `LOG_TABLE`. The absolute error is at most
/// about `3e-6` for every positive `x`, which is close to `f32::ln` but faster (especially in WebAssembly, which has
/// no native logarithm instruction).
///
/// # Arguments
/// * `x` - Positive input value. Values below `f32::MIN_POSITIVE` (including 0) are treated as `f32::MIN_POSITIVE`.
///
/// # Returns
/// * Approximate value of `ln(x)`.
#[inline(always)]
pub fn ln_from_table(x: f32) -> f32 {
    let bits = x.max(f32::MIN_POSITIVE).to_bits();
    let exponent = (bits >> 23) as i32 - 127;
    let mantissa = bits & 0x7f_ffff;
    let index = (mantissa >> FRACTION_BITS) as usize;
    let fraction = (mantissa & ((1 << FRACTION_BITS) - 1)) as f32 / (1 << FRACTION_BITS) as f32;
    exponent as f32 * std::f32::consts::LN_2 + LOG_TABLE[index] + fraction * (LOG_TABLE[index + 1] - LOG_TABLE[index])
}


/// Computes the sum of logarithms for pairs of values in batches to improve performance.
///
/// Each element in `rows` is a `[f32; 2]` pair. The function multiplies elements in
/// small batches (to minimize floating-point underflow) and applies logarithms to
/// the products using `ln_from_table`.
///
/// # Arguments
/// * `rows` - A vector of `[f32; 2]` pairs representing numeric data.
///
/// # Returns
/// * `[f32; 2]` where each component is the batched sum of logarithms across the corresponding column.
pub fn sum_logs_batched(rows: &[[f32; 2]]) -> [f32; 2] {

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;

    let block_size = 10;
    let blocks = rows.len() / block_size;

    for i in 0..blocks {
        let start = i * block_size;
        let mut prod0 = 1.0f32;
        let mut prod1 = 1.0f32;
        for &row in &rows[start..(start+block_size)] {
            prod0 *= row[0];
            prod1 *= row[1];
        }

        acc0 += ln_from_table(prod0);
        acc1 += ln_from_table(prod1);
    }

    let rem_start = blocks * block_size;
    let mut prod0 = 1.0;
    let mut prod1 = 1.0;
    for &row in &rows[rem_start..] {
        prod0 *= row[0];
        prod1 *= row[1];
    }

    [acc0 + ln_from_table(prod0), acc1 + ln_from_table(prod1)]
}


/// Normalizes a vector of floating-point values so that the sum of all elements equals 1.
/// 
/// Mathematically: `x_i = x_i / Σx_j` for all elements `x_i` in the array.
/// 
/// # Arguments
/// * `array` - A mutable reference to a vector of `f32` values.
pub fn normalize(array: &mut [f32]) {
    let sum: f32 = array.iter().sum();
    for val in array.iter_mut() {
        *val /= sum;
    }
}


/// Applies log-normalization to a [f32; 2] using the log-sum-exp trick for stability.
/// 
/// Mathematically: `x_i = exp(x_i - log(Σ exp(x_j)))`
/// 
/// # Arguments
/// * `array` - A mutable reference to a vector of `f32` values.
/// 
/// # Notes
/// - Subtracts `max(x)` to prevent overflow.
pub fn log_normalize(array: &mut [f32; 2]) {
    let max_val = array[0].max(array[1]);
    let log_sum_exp = ((array[0] - max_val).exp() + (array[1] - max_val).exp()).ln();
    array[0] = (array[0] - max_val - log_sum_exp).exp();
    array[1] = (array[1] - max_val - log_sum_exp).exp();
}


/// Prevents numerical underflow by setting a minimum threshold for all elements.
/// 
/// Mathematically: `x_i = max(x_i, 1e-30)`
/// 
/// # Arguments
/// * `array` - A mutable reference to a vector of `f32` values.
pub fn avoid_underflow(array: &mut [f32]) {
    array.iter_mut().for_each(|x| if *x < 1e-30 { *x = 1e-30 });
}


/// Prevents numerical underflow in a fixed-size `[f32; 2]` array by setting a minimum threshold.
///
/// # Arguments
/// * `array` - Mutable reference to a `[f32; 2]` array.
pub fn avoid_underflow_arr(array: &mut [f32; 2]) {
    for item in array.iter_mut().take(2) {
        if *item < 1e-30 {
            *item = 1e-30;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that normalization scales a simple vector to sum to 1.
    #[test]
    fn test_normalize_basic() {
        let mut values = vec![1.0, 1.0, 2.0];
        normalize(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    /// Verifies that log normalization produces a valid probability distribution.
    #[test]
    fn test_log_normalize_basic() {
        let mut values = [0.0, 0.0];
        log_normalize(&mut values);
        let sum: f32 = values[0] + values[1];
        assert!((sum - 1.0).abs() < 1e-12);
    }

    /// Verifies that the table-based logarithm is close to `f64::ln` over the whole range of positive `f32` values,
    /// including very small values and values close to each other.
    #[test]
    fn test_ln_from_table_accuracy() {
        let mut x: f32 = 1e-37;
        while x < 1e3 {
            let expected = (x as f64).ln();
            let error = (ln_from_table(x) as f64 - expected).abs();
            // Allow for the f32 rounding of large results (|ln x| is up to ~85 here).
            assert!(error < 3e-6 + 2e-7 * expected.abs(), "ln_from_table({x}) is off by {error}");
            x *= 1.001;
        }

        // Messages that differ slightly must give different logarithms.
        assert!(ln_from_table(0.5001) > ln_from_table(0.5));
    }

    /// Verifies that the table-based logarithm stays finite for 0 and subnormal inputs.
    #[test]
    fn test_ln_from_table_small_inputs() {
        let expected = f32::MIN_POSITIVE.ln();
        for x in [0.0, 1e-45, 1e-40] {
            assert!((ln_from_table(x) - expected).abs() < 1e-4);
        }
    }

    /// Verifies that underflow protection replaces values below the minimum threshold.
    #[test]
    fn test_avoid_underflow_replaces_small_values() {
        let mut values = vec![1e-40, 1e-20];
        avoid_underflow(&mut values);
        assert!(values[0] >= 1e-30);
        assert!(values[1] >= 1e-30);
    }
}