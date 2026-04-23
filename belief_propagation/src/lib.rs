extern crate serde_json;
extern crate serde;

mod array_utils;
mod convolution_tree;
mod factor_graph;
mod messages;
mod node;
mod zero_lookahead_belief_propagation;

use crate::{factor_graph::CTFactorGraph, zero_lookahead_belief_propagation::{run_belief_propagation, generate_csv, calibrate_all_subgraphs}};


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
    alpha: f32,
    beta: f32,
    regularized: bool,
    prior: f32,
    max_iter: Option<u32>,
    tolerance: Option<f32>
) -> String {
    let max_iter: u32 = max_iter.unwrap_or(10000);
    let tolerance: f32 = tolerance.unwrap_or(0.006);
    
    run_belief_propagation(graph, alpha, beta, regularized, prior, max_iter, tolerance).unwrap()
}


/// Load factor graph provided as a GraphML string.
///
/// This function constructs the factor graph, fills in factor tables and priors and
/// splits the graph into connected components.
///
/// # Arguments
///
/// * `graph` - GraphML representation of the factor graph.
/// * `alpha` - Noisy-OR factor alpha parameter.
/// * `beta` - Noisy-OR factor beta parameter.
/// * `regularized` - Whether to regularize factor tables to penalize large numbers of parents.
/// * `prior` - Prior belief for taxon nodes.
///
/// # Returns
///
/// Vector of graphs
pub fn load_factor_graph(
    graph: String,
) -> Result<Vec<CTFactorGraph>, Box<dyn std::error::Error>> {
    let mut ct_factor_graph = CTFactorGraph::from_graphml(&graph)?;

    ct_factor_graph.add_ct_nodes();

    let ct_factor_graphs: Vec<CTFactorGraph> = ct_factor_graph.connected_components();

    Ok(ct_factor_graphs)
}


/// Runs belief propagation on a factor graph.
///
/// This function performs loopy belief propagation
/// on each component. The result is returned as a CSV string.
///
/// # Arguments
///
/// * `graphs` - Vector of graphs.
/// * `max_iter` - Maximum number of belief propagation iterations.
/// * `tol` - Tolerance threshold for message convergence.
///
/// # Returns
///
/// CSV string with one row per node containing columns:
/// `[node_name, posterior_probability_1, node_category]`
pub fn zero_lookahead_bp_from_graph(
    graphs: &mut Vec<CTFactorGraph>,
    alpha: f32,
    beta: f32,
    regularized: bool,
    prior: f32,
    max_iter: Option<u32>,
    tolerance: Option<f32>
) -> Result<String, Box<dyn std::error::Error>> {
    let max_iter: u32 = max_iter.unwrap_or(10000);
    let tolerance: f32 = tolerance.unwrap_or(0.006);

    for ct_factor_graph in graphs.iter_mut() {
        ct_factor_graph.fill_in_factors(alpha, beta, regularized);
        ct_factor_graph.fill_in_priors(prior);
    }

    let results: Vec<(String, Vec<f32>)> = calibrate_all_subgraphs(
        graphs,
        max_iter,
        tolerance
    )?;

    Ok(generate_csv(results)?)
}