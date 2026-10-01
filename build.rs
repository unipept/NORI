use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Number of mantissa bits used to index the log table: the table has `2^LOG_TABLE_BITS + 1` entries.
const LOG_TABLE_BITS: u32 = 10;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let path = Path::new(&out_dir).join("log_table.rs");
    let mut file = File::create(&path).unwrap();

    // LOG_TABLE[i] = ln(1 + i / 2^LOG_TABLE_BITS): ln of a float mantissa at evenly spaced points in [1, 2].
    // The last entry (ln 2) lets `ln_from_table` interpolate in the last interval without a bounds check.
    let size = 1usize << LOG_TABLE_BITS;
    writeln!(file, "pub const LOG_TABLE_BITS: u32 = {LOG_TABLE_BITS};").unwrap();
    // The last entry is ln 2, which clippy would otherwise flag as an approximation of `LN_2`.
    writeln!(file, "#[allow(clippy::approx_constant)]").unwrap();
    writeln!(file, "pub const LOG_TABLE: [f32; {}] = [", size + 1).unwrap();
    for i in 0..=size {
        let mantissa = 1.0 + i as f64 / size as f64;
        writeln!(file, "    {:?},", mantissa.ln() as f32).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Tell Cargo to rerun build.rs if this file changes (optional)
    println!("cargo:rerun-if-changed=build.rs");
}
