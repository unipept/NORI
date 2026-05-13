
use std::fs;
use nori::{load_factor_graph, zero_lookahead_bp_from_graph};

fn benchmark() -> std::io::Result<()> {
    println!("Benchmark: loading graph from '../input_data/iPRG2016/peptide_protein_B.graphml'...");
    let input = fs::read_to_string("../input_data/iPRG2016/peptide_protein_B.graphml")
        .expect("Failed to read graph file");

    let priors = [0.2, 0.5, 0.7];
    let betas = [0.01, 0.2, 0.4];
    let alphas = [0.1, 0.25, 0.5, 0.65, 0.8];

    let mut graphs_template = load_factor_graph(&input).unwrap();
    println!(
        "Benchmark: running zero-lookahead belief propagation with {} alpha x {} beta x {} prior combinations...",
        alphas.len(), betas.len(), priors.len()
    );

    let total_runs = alphas.len() * betas.len() * priors.len();
    let mut run_count = 0;

    for &alpha in &alphas {
        for &beta in &betas {
            for &prior in &priors {
                run_count += 1;
                println!(
                    "\n=== Run {}/{} ===\nalpha = {}\nbeta = {}\nprior = {}\n",
                    run_count, total_runs, alpha, beta, prior
                );

                // let mut graphs = graphs_template.clone();
                let results = zero_lookahead_bp_from_graph(
                    &mut graphs_template,
                    alpha,
                    beta,
                    true,
                    prior,
                    Some(10000),
                    Some(0.005),
                );
                println!("Result CSV output:\n{:?}", results.unwrap());
            }
        }
    }

    println!("\nBenchmark complete: {} runs finished.", total_runs);
    Ok(())
}


fn main() {
    let _ = benchmark();
}