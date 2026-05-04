
include!(concat!(env!("OUT_DIR"), "/log_table.rs"));

const N: usize = 1024;
const MIN_X: f32 = 1e-10;
const MAX_X: f32 = 1.0;
const STEP: f32 = (MAX_X - MIN_X) / (N as f32 - 1.0);


/// Approximates the natural logarithm of `x` using a precomputed lookup table.
///
/// # Arguments
/// * `x` - Input value between `MIN_X` and `MAX_X`.
///
/// # Returns
/// * Approximate value of `ln(x)` retrieved from a static lookup table.
#[inline(always)]
pub fn ln_from_table(x: f32) -> f32 {
    let idx = ((x - MIN_X) / STEP) as usize;
    LOG_TABLE[idx] as f32
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
pub fn sum_logs_batched(rows: &Vec<[f32; 2]>) -> [f32; 2] {

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
pub fn normalize(array: &mut Vec<f32>) {
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
pub fn avoid_underflow(array: &mut Vec<f32>) {
    array.iter_mut().for_each(|x| if *x < 1e-30 { *x = 1e-30 });
}


/// Prevents numerical underflow in a fixed-size `[f32; 2]` array by setting a minimum threshold.
///
/// # Arguments
/// * `array` - Mutable reference to a `[f32; 2]` array.
pub fn avoid_underflow_arr(array: &mut [f32; 2]) {
    for i in 0..2 {
        if array[i] < 1e-30 {
            array[i] = 1e-30;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let mut values = vec![1.0, 1.0, 2.0];
        normalize(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_log_normalize_basic() {
        let mut values = [0.0, 0.0];
        log_normalize(&mut values);
        let sum: f32 = values[0] + values[1];
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_avoid_underflow_replaces_small_values() {
        let mut values = vec![1e-40, 1e-20];
        avoid_underflow(&mut values);
        assert!(values[0] >= 1e-30);
        assert!(values[1] >= 1e-30);
    }
}