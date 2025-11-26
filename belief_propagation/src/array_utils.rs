
include!(concat!(env!("OUT_DIR"), "/log_table.rs"));

const N: usize = 1024;
const MIN_X: f64 = 1e-10;
const MAX_X: f64 = 1.0;
const STEP: f64 = (MAX_X - MIN_X) / (N as f64 - 1.0);


/// Approximates the natural logarithm of `x` using a precomputed lookup table.
///
/// # Arguments
/// * `x` - Input value between `MIN_X` and `MAX_X`.
///
/// # Returns
/// * Approximate value of `ln(x)` retrieved from a static lookup table.
#[inline(always)]
pub fn ln_from_table(x: f64) -> f64 {
    let idx = ((x - MIN_X) / STEP) as usize;
    LOG_TABLE[idx]
}


/// Computes the sum of logarithms for pairs of values in batches to improve performance.
///
/// Each element in `rows` is a `[f64; 2]` pair. The function multiplies elements in
/// small batches (to minimize floating-point underflow) and applies logarithms to
/// the products using `ln_from_table`.
///
/// # Arguments
/// * `rows` - A vector of `[f64; 2]` pairs representing numeric data.
///
/// # Returns
/// * `[f64; 2]` where each component is the batched sum of logarithms across the corresponding column.
pub fn sum_logs_batched(beliefs: &Vec<f64>, belief_length: usize) -> Vec<f64> {

    let mut accs = vec![0.0; belief_length];

    let block_size = 6;
    let blocks = beliefs.len() / (block_size * belief_length);

    for block_id in 0..blocks {
        let block_start = block_id * belief_length * block_size;
        let mut prods: Vec<f64> = vec![1.0; belief_length];

        for row_id in 0..block_size {
            let row_start = block_start + row_id * belief_length;

            for i in 0..belief_length {
                prods[i] *= beliefs[row_start + i];
            }
        }

        for i in 0..belief_length {
            accs[i] += ln_from_table(prods[i]);
        }
    }

    let mut rem_start = blocks * block_size * belief_length;
    let mut prods = vec![1.0; belief_length];
    while rem_start < beliefs.len() {
        for i in 0..belief_length {
            prods[i] *= beliefs[rem_start + i];
        }
        rem_start += belief_length;
    }

    for i in 0..belief_length {
        accs[i] += ln_from_table(prods[i]);
    }

    accs
}


/// Normalizes a vector of floating-point values so that the sum of all elements equals 1.
/// 
/// Mathematically: `x_i = x_i / Σx_j` for all elements `x_i` in the array.
/// 
/// # Arguments
/// * `array` - A mutable reference to a vector of `f64` values.
pub fn normalize(array: &mut Vec<f64>) {
    normalize_slice(array, 0, array.len());
}


pub fn normalize_slice(array: &mut Vec<f64>, start: usize, end: usize) {
    let sum: f64 = array[start..end].iter().sum();
    for val in array[start..end].iter_mut() {
        *val /= sum;
    }
}


/// Applies log-normalization to a [f64; 2] using the log-sum-exp trick for stability.
/// 
/// Mathematically: `x_i = exp(x_i - log(Σ exp(x_j)))`
/// 
/// # Arguments
/// * `array` - A mutable reference to a vector of `f64` values.
/// 
/// # Notes
/// - Subtracts `max(x)` to prevent overflow.
pub fn log_normalize(array: &mut Vec<f64>) {
    let max_val = array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let log_sum_exp = array
        .iter()
        .map(|&x| (x - max_val).exp())
        .sum::<f64>()
        .ln();

    for x in array.iter_mut() {
        *x = ( *x - max_val - log_sum_exp ).exp();
    }
}


/// Prevents numerical underflow by setting a minimum threshold for all elements.
/// 
/// Mathematically: `x_i = max(x_i, 1e-30)`
/// 
/// # Arguments
/// * `array` - A mutable reference to a vector of `f64` values.
pub fn avoid_underflow(array: &mut Vec<f64>) {
    array.iter_mut().for_each(|x| if *x < 1e-10 { *x = 1e-10 });
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let mut values = vec![1.0, 1.0, 2.0];
        normalize(&mut values);
        let sum: f64 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_2d_basic() {
        let mut values = vec![[1.0, 1.0], [2.0, 2.0]];
        normalize_2d(&mut values);
        let total: f64 = values.iter().map(|x| x[0] + x[1]).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_log_normalize_basic() {
        let mut values = [0.0, 0.0];
        log_normalize(&mut values);
        let sum: f64 = values[0] + values[1];
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_log_normalize_2d_basic() {
        let mut values = vec![[0.0, 0.0], [1.0, 1.0]];
        log_normalize_2d(&mut values);
        let total: f64 = values.iter().map(|x| x[0] + x[1]).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_avoid_underflow_replaces_small_values() {
        let mut values = vec![1e-40, 1e-20];
        avoid_underflow(&mut values);
        assert!(values[0] >= 1e-30);
        assert!(values[1] >= 1e-30);
    }
}