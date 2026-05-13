use crate::factor_graph::CTFactorGraph;
use crate::messages::Messages;

/// Type alias for belief propagation results.
type BeliefResult = Vec<(String, Vec<f32>)>;


/// Calibrates multiple subgraphs (connected components) of a factor graph using loopy belief propagation.
///
/// # Arguments
///
/// * `ct_factor_graphs` - Vector of `CTFactorGraph` objects representing connected subgraphs of the full factor graph.
/// * `max_iterations` - Maximum number of iterations for message passing in case of non-convergence.
/// * `tolerance` - Convergence criterion; the maximum allowable change in messages between iterations.
///
/// # Returns
///
/// A vector of belief results for all output nodes in connected components.
/// Each element is a tuple `(node_name, belief_distribution)` where `belief_distribution` is `[P(0), P(1)]`.
pub fn calibrate_all_subgraphs(
    ct_factor_graphs: &Vec<CTFactorGraph>,
    max_iterations: u32,
    tolerance: f32
) -> Result<BeliefResult, Box<dyn std::error::Error>>{
    let mut results: Vec<(String, Vec<f32>)> = Vec::new();

    for subgraph in ct_factor_graphs {
        if subgraph.node_count() > 2 {

            let mut messages = Messages::new(subgraph);
            let beliefs: Vec<(String, Vec<f32>)> = messages.zero_lookahead_bp(
                max_iterations,
                tolerance
            )?;

            results.extend(beliefs);
        }
    }

    Ok(results)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Ensures calibrate_all_subgraphs returns an empty result for no input graphs.
    #[test]
    fn test_calibrate_all_subgraphs_empty() {
        let res = calibrate_all_subgraphs(&vec![], 10, 1e-6);
        assert!(res.is_ok());
        let res = res.unwrap();

        assert!(res.is_empty());
    }
}
