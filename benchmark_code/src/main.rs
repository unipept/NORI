
use std::fs;
use infernor::{load_factor_graph, zero_lookahead_bp_from_graph};

fn benchmark() -> std::io::Result<()> {
    let input = fs::read_to_string("../input_data/iPRG2016/peptide_protein_B.graphml")
        .expect("Failed to read graph file");

    let priors = [0.2, 0.5, 0.7];
    let betas = [0.01, 0.2, 0.4];
    let alphas = [0.1, 0.25, 0.5, 0.65, 0.8];

    let mut graphs = load_factor_graph(input).unwrap();

    for &alpha in &alphas {
        for &beta in &betas {
            for &prior in &priors {
                let csv = zero_lookahead_bp_from_graph(
                    &mut graphs,
                    alpha,
                    beta,
                    true,
                    prior,
                    Some(10000),
                    Some(0.005),
                );
                println!("{}", csv.unwrap())
            }
        }
    }

    Ok(())
}


fn main() {
    let _ = benchmark();
}