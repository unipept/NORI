extern crate serde;

mod array_utils;
mod convolution_tree;
mod factor_graph;
mod messages;
mod node;
mod zero_lookahead_belief_propagation;

use crate::{factor_graph::CTFactorGraph, zero_lookahead_belief_propagation::calibrate_all_subgraphs};

/// Type alias for belief propagation results.
type BeliefResult = Vec<(String, Vec<f32>)>;

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
/// * `prior` - Prior belief for output variable nodes.
/// * `max_iter` - Maximum number of belief propagation iterations.
/// * `tolerance` - Tolerance threshold for message convergence.
///
/// # Returns
///
/// Vector of tuples containing node names and their belief distributions.
pub fn zero_lookahead_bp(
    graph: String,
    alpha: f32,
    beta: f32,
    regularized: bool,
    prior: f32,
    max_iter: Option<u32>,
    tolerance: Option<f32>
) -> Result<BeliefResult, Box<dyn std::error::Error>> {
    let max_iter: u32 = max_iter.unwrap_or(10000);
    let tolerance: f32 = tolerance.unwrap_or(0.006);
    
    let mut ct_factor_graph = CTFactorGraph::from_graphml(&graph)?;
    ct_factor_graph.fill_in_factors(alpha, beta, regularized);
    ct_factor_graph.fill_in_priors(prior);
    ct_factor_graph.add_ct_nodes();
    let ct_factor_graphs: Vec<CTFactorGraph> = ct_factor_graph.connected_components();

    let results = calibrate_all_subgraphs(
        &ct_factor_graphs,
        max_iter,
        tolerance
    )?;

    Ok(results)
}


/// Load factor graph provided as a GraphML string.
///
/// This function parses the GraphML into a factor graph, adds convolution tree nodes,
/// and splits the graph into connected components.
///
/// # Arguments
///
/// * `graph` - GraphML representation of the factor graph.
///
/// # Returns
///
/// A vector of `CTFactorGraph` objects representing connected subgraphs.
pub fn load_factor_graph(
    graph: &str,
) -> Result<Vec<CTFactorGraph>, Box<dyn std::error::Error>> {
    let mut ct_factor_graph = CTFactorGraph::from_graphml(graph)?;

    ct_factor_graph.add_ct_nodes();

    let ct_factor_graphs: Vec<CTFactorGraph> = ct_factor_graph.connected_components();

    Ok(ct_factor_graphs)
}

/// Serializes a factor graph to a byte array for storage or transmission.
///
/// This function loads a factor graph from a GraphML string, adds convolution tree nodes,
/// splits into connected components, and serializes the result to bytes using bincode.
/// The serialized format can be deserialized and used with `zero_lookahead_bp_from_graph_bytes`.
///
/// # Arguments
///
/// * `graph` - GraphML representation of the factor graph.
///
/// # Returns
///
/// A byte vector containing the serialized graph components, or an error if serialization fails.
pub fn load_factor_graph_bytes(
    graph: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let ct_factor_graphs = load_factor_graph(graph)?;

    let bytes: Vec<u8> = bincode::serialize(&ct_factor_graphs)?;
    
    Ok(bytes)
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
/// Vector of tuples containing node names and their belief distributions.
pub fn zero_lookahead_bp_from_graph(
    graphs: &mut Vec<CTFactorGraph>,
    alpha: f32,
    beta: f32,
    regularized: bool,
    prior: f32,
    max_iter: Option<u32>,
    tolerance: Option<f32>
) -> Result<BeliefResult, Box<dyn std::error::Error>> {
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

    Ok(results)
}

/// Runs belief propagation on a factor graph deserialized from bytes.
///
/// This function deserializes a graph from bytes (previously created by `load_factor_graph_bytes`),
/// fills in factor tables and priors, and performs loopy belief propagation on all components.
/// This is more efficient than `zero_lookahead_bp` when the graph is already available in serialized form.
///
/// # Arguments
///
/// * `graph_bytes` - Serialized graph bytes (from `load_factor_graph_bytes`).
/// * `alpha` - Noisy-OR factor alpha parameter.
/// * `beta` - Noisy-OR factor beta parameter.
/// * `regularized` - Whether to regularize factor tables to penalize large numbers of parents.
/// * `prior` - Prior belief for output variable nodes.
/// * `max_iter` - Maximum number of belief propagation iterations. Defaults to 10,000 if `None`.
/// * `tolerance` - Tolerance threshold for message convergence. Defaults to 0.006 if `None`.
///
/// # Returns
///
/// Vector of tuples containing node names and their belief distributions,
/// or an error if deserialization or belief propagation fails.
pub fn zero_lookahead_bp_from_graph_bytes(
    graph_bytes: &[u8],
    alpha: f32,
    beta: f32,
    regularized: bool,
    prior: f32,
    max_iter: Option<u32>,
    tolerance: Option<f32>
) -> Result<BeliefResult, Box<dyn std::error::Error>> {
    let mut graphs: Vec<CTFactorGraph> = bincode::deserialize(graph_bytes).unwrap();
    
    zero_lookahead_bp_from_graph(
        &mut graphs,
        alpha,
        beta,
        regularized,
        prior,
        max_iter,
        tolerance
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: f32 = 0.9;
    const BETA: f32 = 0.7;
    const SCORE: f32 = 0.7;

    /// Builds a GraphML string with one input node per peptide (belief `[1 - SCORE, SCORE]`), one output node per
    /// parent, and an edge for every `(peptide, parent)` pair.
    fn graphml(edges: &[(&str, &str)]) -> String {
        let mut peptides: Vec<&str> = edges.iter().map(|edge| edge.0).collect();
        peptides.sort();
        peptides.dedup();
        let mut parents: Vec<&str> = edges.iter().map(|edge| edge.1).collect();
        parents.sort();
        parents.dedup();

        let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><key id="type" for="node"/><key id="belief" for="node"/><graph edgedefault="undirected">"#);
        for peptide in peptides {
            xml += &format!(r#"<node id="{peptide}"><data key="type">input</data><data key="belief">[{}, {SCORE}]</data></node>"#, 1.0 - SCORE);
        }
        for parent in parents {
            xml += &format!(r#"<node id="{parent}"><data key="type">output</data></node>"#);
        }
        for (peptide, parent) in edges {
            xml += &format!(r#"<edge source="{peptide}" target="{parent}"/>"#);
        }
        xml + "</graph></graphml>"
    }

    /// Computes the exact posterior P(parent = 1) of every parent by enumerating all parent states of the
    /// (unregularized) noisy-OR model.
    fn exact_posteriors(edges: &[(&str, &str)], prior: f64) -> Vec<(String, f64)> {
        let mut parents: Vec<&str> = edges.iter().map(|edge| edge.1).collect();
        parents.sort();
        parents.dedup();
        let mut peptides: Vec<&str> = edges.iter().map(|edge| edge.0).collect();
        peptides.sort();
        peptides.dedup();
        let parents_of: Vec<Vec<usize>> = peptides.iter()
            .map(|peptide| edges.iter()
                .filter(|edge| edge.0 == *peptide)
                .map(|edge| parents.iter().position(|parent| *parent == edge.1).unwrap())
                .collect())
            .collect();

        let (alpha, beta, score) = (ALPHA as f64, BETA as f64, SCORE as f64);
        let mut evidence = 0.0;
        let mut marginals = vec![0.0; parents.len()];
        for state in 0..(1usize << parents.len()) {
            let active = |parent: usize| state >> parent & 1 == 1;
            let active_count = (0..parents.len()).filter(|&parent| active(parent)).count() as i32;
            let mut probability = prior.powi(active_count) * (1.0 - prior).powi(parents.len() as i32 - active_count);
            for peptide_parents in &parents_of {
                let k = peptide_parents.iter().filter(|&&parent| active(parent)).count() as i32;
                let present = 1.0 - (1.0 - alpha).powi(k) * (1.0 - beta);
                probability *= score * present + (1.0 - score) * (1.0 - present);
            }
            evidence += probability;
            for (parent, marginal) in marginals.iter_mut().enumerate() {
                if active(parent) {
                    *marginal += probability;
                }
            }
        }

        parents.iter().zip(marginals).map(|(parent, marginal)| (parent.to_string(), marginal / evidence)).collect()
    }

    /// Runs unregularized belief propagation on `edges` and checks every posterior against exact enumeration.
    fn assert_matches_exact(edges: &[(&str, &str)], prior: f32, tolerance: f64) {
        let beliefs = zero_lookahead_bp(graphml(edges), ALPHA, BETA, false, prior, None, None).unwrap();
        for (parent, expected) in exact_posteriors(edges, prior as f64) {
            let (_, belief) = beliefs.iter().find(|(name, _)| *name == parent).unwrap();
            assert!(
                (belief[1] as f64 - expected).abs() < tolerance,
                "{parent}: belief propagation gave {}, exact posterior is {expected}", belief[1]
            );
        }
    }

    /// A peptide shared by several parents is handled by a convolution tree, whose messages must reach the parents.
    #[test]
    fn test_shared_peptide_matches_exact() {
        for prior in [0.3, 0.01] {
            assert_matches_exact(&[("P1", "F1"), ("P1", "F2")], prior, 1e-4);
            assert_matches_exact(&[("P1", "F1"), ("P1", "F2"), ("P1", "F3"), ("P1", "F4"), ("P1", "F5")], prior, 1e-4);
        }
    }

    /// Several convolution trees in one tree-shaped graph.
    #[test]
    fn test_multiple_shared_peptides_matches_exact() {
        let edges = [("P1", "F1"), ("P1", "F2"), ("P2", "F3"), ("P2", "F4"), ("P2", "F5"), ("P3", "F6")];
        for prior in [0.3, 0.01] {
            assert_matches_exact(&edges, prior, 1e-4);
        }
    }
}
