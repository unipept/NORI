use crate::factor_graph::CTFactorGraph;
use crate::messages::Messages;
use csv::Writer;


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
/// Tuple `(node_names, node_categories, results)`:
/// * `node_names` - Vector of node names in the same order as the belief results.
/// * `node_categories` - Vector of node types (categories) corresponding to the nodes.
/// * `results` - Vector of belief distributions for each node; each element is a vector `[P(0), P(1)]`.
pub fn calibrate_all_subgraphs(
    ct_factor_graphs: &Vec<CTFactorGraph>,
    max_iterations: u32,
    tolerance: f64
) -> Result<Vec<(String, Vec<f64>)>, Box<dyn std::error::Error>>{
    let mut results: Vec<(String, Vec<f64>)> = Vec::new();

    for subgraph in ct_factor_graphs {
        if subgraph.node_count() > 2 {

            let mut messages = Messages::new(subgraph);
            let beliefs: Vec<(String, Vec<f64>)> = messages.zero_lookahead_bp(
                max_iterations,
                tolerance
            )?;

            results.extend(beliefs);
        }
    }

    Ok(results)
}


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
pub fn run_belief_propagation(
    graph: String,
    alpha: f64,
    beta: f64,
    regularized: bool,
    prior: f64,
    max_iter: u32,
    tol: f64
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ct_factor_graph = CTFactorGraph::from_graphml(&graph)?;
    ct_factor_graph.fill_in_factors(alpha, beta, regularized);
    ct_factor_graph.fill_in_priors(prior);
    ct_factor_graph.add_ct_nodes();
    let ct_factor_graphs: Vec<CTFactorGraph> = ct_factor_graph.connected_components();

    let results = calibrate_all_subgraphs(
        &ct_factor_graphs,
        max_iter,
        tol
    )?;

    Ok(generate_csv(results)?)
}


/// Generates a CSV string from node names, types, and belief results.
///
/// # Arguments
///
/// * `node_names` - Vector of node names.
/// * `results` - Vector of belief distributions for each node; each element is a vector `[P(0), P(1)]`.
///
/// # Returns
///
/// CSV string with columns `[node_name, posterior_probability_1, node_category]`.
pub fn generate_csv(results: Vec<(String, Vec<f64>)>) -> Result<String, Box<dyn std::error::Error>> {

    let mut wtr = Writer::from_writer(vec![]);

    for (node_name, belief) in results {
        
        let _ = wtr.write_record(&[
            node_name,
            format!("[{}]", belief.iter()
                .map(|b| b.to_string()).collect::<Vec<_>>().join(",")),
        ])?;
    }

    let csv: String = String::from_utf8(wtr.into_inner()?)?;

    Ok(csv)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_csv_basic() {
        let results = vec![("n1".to_string(), vec![0.3, 0.7]), ("n2".to_string(), vec![0.6, 0.4])];

        let csv = generate_csv(results);
        assert!(csv.is_ok());
        let csv = csv.unwrap();

        assert!(csv.contains("n1"));
        assert!(csv.contains("0.7"));
    }

    #[test]
    fn test_calibrate_all_subgraphs_empty() {
        let res = calibrate_all_subgraphs(&vec![], 10, 1e-6);
        assert!(res.is_ok());
        let res = res.unwrap();

        assert!(res.is_empty());
    }

    #[test]
    fn test_run_belief_propagation_does_not_crash() {
        let minimal_graph = r#"<?xml version='1.0' encoding='utf-8'?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
            <key id="type" for="node" attr.name="node_type" attr.type="string"/>
            <key id="belief" for="node" attr.name="belief" attr.type="string"/>
            <graph edgedefault="undirected">
                <node id="n0">
                    <data key="type">input</data>
                    <data key="belief">[0.00437876, 0.99562124]</data>
                </node>
                <node id="n1">
                    <data key="type">output</data>
                </node>
                <edge id="e1" source="n0" target="n1"/>
            </graph>
        </graphml>
        "#.to_string();

        let csv = run_belief_propagation(
            minimal_graph,
            0.5,   // alpha
            0.5,   // beta
            true,  // regularized
            0.1,   // prior
            10,    // max_iter
            1e-6   // tol
        );
        assert!(csv.is_ok());
        let csv = csv.unwrap();

        assert!(csv.contains("n1"));
    }
}
