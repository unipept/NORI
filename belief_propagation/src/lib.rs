extern crate serde_json;
extern crate serde;

mod array_utils;
mod convolution_tree;
mod factor_graph;
mod messages;
mod node;
mod zero_lookahead_belief_propagation;

use crate::zero_lookahead_belief_propagation::run_belief_propagation;


/// Runs belief propagation on a factor graph provided as a GraphML string.
///
/// This function constructs the factor graph, fills in factor tables and priors,
/// splits the graph into connected components, and performs loopy belief propagation
/// on each component. The result is returned as a CSV string.
///
/// # Arguments
///
/// * `graph` - GraphML representation of the factor graph.
/// * `alpha` - Noisy-OR factor alpha parameter.
/// * `beta` - Noisy-OR factor beta parameter.
/// * `regularized` - Whether to regularize factor tables to penalize large numbers of parents.
/// * `prior` - Prior belief for taxon nodes.
/// * `max_iter` - Maximum number of belief propagation iterations.
/// * `tol` - Tolerance threshold for message convergence.
///
/// # Returns
///
/// CSV string with one row per node containing columns:
/// `[node_name, posterior_probability_1, node_category]`
pub fn zero_lookahead_bp(
    graph: String,
    alpha: f64,
    beta: f64,
    regularized: bool,
    prior: f64,
    max_iter: Option<u32>,
    tolerance: Option<f64>
) -> String {
    let max_iter: u32 = max_iter.unwrap_or(10000);
    let tolerance: f64 = tolerance.unwrap_or(0.006);
    
    run_belief_propagation(graph, alpha, beta, regularized, prior, max_iter, tolerance).unwrap()
}