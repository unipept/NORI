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
fn calibrate_all_subgraphs(
    ct_factor_graphs: Vec<CTFactorGraph>,
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
        ct_factor_graphs,
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
fn generate_csv(results: Vec<(String, Vec<f64>)>) -> Result<String, Box<dyn std::error::Error>> {

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
    use std::collections::HashMap;

    #[test]
    fn test_generate_csv_basic() {
        let node_names = vec!["n1".to_string(), "n2".to_string()];
        let node_types = vec!["taxon".to_string(), "peptide".to_string()];
        let results = vec![vec![0.3, 0.7], vec![0.6, 0.4]];

        let csv = generate_csv(node_names, node_types, results);
        assert!(csv.is_ok());
        let csv = csv.unwrap();

        assert!(csv.contains("n1"));
        assert!(csv.contains("0.7"));
        assert!(csv.contains("taxon"));
    }

    #[test]
    fn test_parse_taxon_scores_basic() {
        let csv_content = "123,0.8,taxon\n456,0.5,taxon\n789,0.9,peptide\n".to_string();
        let json = parse_taxon_scores(csv_content);
        assert!(json.is_ok());
        let json = json.unwrap();

        let parsed: HashMap<usize, f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.get(&456), Some(&0.5));
        assert_eq!(parsed.get(&123), Some(&0.8));
        assert!(parsed.get(&789).is_none());
    }

    #[test]
    fn test_calibrate_all_subgraphs_empty() {
        let res = calibrate_all_subgraphs(vec![], 10, 1e-6);
        assert!(res.is_ok());
        let (names, cats, results) = res.unwrap();

        assert!(names.is_empty());
        assert!(cats.is_empty());
        assert!(results.is_empty());
    }

    #[test]
    fn test_run_belief_propagation_does_not_crash() {
        let minimal_graph = r#"<?xml version='1.0' encoding='utf-8'?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
            <key id="d1" for="node" attr.name="InitialBelief_1" attr.type="double"/>
            <key id="d0" for="node" attr.name="InitialBelief_0" attr.type="double"/>
            <graph edgedefault="undirected">
                <node id="n0">
                    <data key="d0">0.0010000000000000009</data>
                    <data key="d1">0.999</data>
                    <data key="d2">peptide</data>
                </node>
                <node id="n1">
                    <data key="d2">factor</data>
                    <data key="d3">2</data>
                </node>
                <node id="n2">
                    <data key="d2">taxon</data>
                </node>
                <edge source="n0" target="n1"/>
                <edge source="n1" target="n2"/>
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
        println!("{}", csv);

        assert!(csv.contains("n0"));
    }
}
