use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::convert::TryInto;


/// Defines the type of node in the factor graph with its initial beliefs.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum NodeType {
    /// Variable node with prior probabilities.
    Variable { output: bool, name: String, initial_belief: [f32; 2] },
    /// Factor node with CPD.
    Factor { initial_belief: Vec<[f32; 2]> },
    /// Convolution tree node.
    ConvolutionTree
}


/// Represents a node in the factor graph with its attributes and connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    id: u32,
    incident_edges: Vec<u32>,
    subtype: NodeType
}


impl Node {

    /// Creates a new node with given ID, and subtype.
    ///
    /// # Arguments
    /// * `id` - Unique node identifier.
    /// * `subtype` - Type of the node with initial state.
    ///
    /// # Returns
    /// A new `Node`.
    pub fn new(id: usize, subtype: NodeType) -> Self {
        let incident_edges: Vec<u32> = Vec::new();
     
        Self { id: id as u32, incident_edges, subtype }
    }

    /// Returns the variable node name, or an error if the node is not a variable.
    pub fn get_name(&self) -> Result<&str, Box<dyn std::error::Error>> {
        if let NodeType::Variable { name, .. } = self.get_subtype() {
            return Ok(name);
        }
        Err("get_name called on a factor/convolution node".into())
    }

    /// Creates a copy of the node with a new ID.
    ///
    /// # Arguments
    /// * `new_id` - Replacement node ID.
    ///
    /// # Returns
    /// A cloned node with updated ID.
    pub fn copy_with_id(&self, new_id: usize) -> Self {
        let mut copy: Node = self.clone();
        copy.id = new_id as u32;
        copy
    }

    /// Creates a new convolution tree node.
    ///
    /// # Arguments
    /// * `id` - Node ID.
    ///
    /// # Returns
    /// A new convolution tree node.
    pub fn new_convolution_node(id: usize) -> Self {
        Self { id: id as u32, incident_edges: Vec::new(), subtype: NodeType::ConvolutionTree }
    }

    /// Adds an incident edge to the node.
    ///
    /// # Arguments
    /// * `edge` - Edge identifier.
    pub fn add_incident_edge(&mut self, edge: usize) {
        self.incident_edges.push(edge as u32);
    }

    /// Returns the node's ID.
    pub fn get_id(&self) -> usize {
        self.id as usize
    }

    /// Returns a reference to the node subtype.
    pub fn get_subtype(&self) -> &NodeType {
        &self.subtype
    }

    /// Updates the node subtype.
    ///
    /// # Arguments
    /// * `subtype` - New subtype for the node.
    pub fn set_subtype(&mut self, subtype: NodeType) {
        self.subtype = subtype;
    }

    /// Returns the number of neighbors of the node.
    pub fn neighbors_count(&self) -> usize {
        self.incident_edges.len()
    }

    /// Returns a specific incident edge by neighbor index within the nodes neighbors.
    ///
    /// # Arguments
    /// * `neighbor_id` - Index of neighbor.
    ///
    /// # Returns
    /// Edge identifier.
    pub fn get_incident_edge(&self, neighbor_id: usize) -> usize {
        self.incident_edges[neighbor_id] as usize
    }

    /// Returns all incident edges.
    pub fn get_incident_edges(&self) -> impl Iterator<Item = usize> + use<'_> {
        self.incident_edges.iter().map(|&edge_id| edge_id as usize)
    }

    /// Replaces incident edges with a new list.
    ///
    /// # Arguments
    /// * `new_incident_edges` - Replacement edge list.
    pub fn set_incident_edges(&mut self, new_incident_edges: impl Iterator<Item = usize>) {
        self.incident_edges = new_incident_edges.map(|x| x as u32).collect();
    }

    /// Checks if the node is a factor node.
    pub fn is_factor_node(&self) -> bool {
        matches!(self.subtype, NodeType::Factor { .. })
    }

    /// Checks if the node is an output node.
    pub fn is_output_node(&self) -> bool {
        matches!(self.subtype, NodeType::Variable { output: true , .. })
    }

    /// Checks if the node is an input node.
    pub fn is_input_node(&self) -> bool {
        matches!(self.subtype, NodeType::Variable { output: false , .. })
    }

    /// Checks if the node is a convolution node.
    pub fn is_convolution_tree_node(&self) -> bool {
        matches!(self.subtype, NodeType::ConvolutionTree)
    }

    /// Updates prior belief for variable nodes.
    ///
    /// # Arguments
    /// * `prior` - Probability to assign as active.
    pub fn fill_in_prior(&mut self, prior: f32) {
        if let NodeType::Variable { output: true , name, .. } = &self.subtype {
            self.subtype = NodeType::Variable { output: true, name: name.to_string(), initial_belief: [1.0 - prior, prior] };
        }
    }

    /// Initializes factor CPD with noisy-OR parameters.
    ///
    /// # Arguments
    /// * `alpha` - Input activation probability.
    /// * `beta` - Noise parameter.
    /// * `regularized` - Whether to apply parent-count regularization.
    pub fn fill_in_factor(&mut self, degree: usize, alpha: f32, beta: f32, regularized: bool) {
        let mut cpd_array: Vec<[f32; 2]> = Vec::with_capacity(degree);
        let mut cpd_array_regularized = Vec::with_capacity(degree);
        let exponent_array: Vec<usize> = (0..degree).collect();
        let divide_array: Vec<f32> = std::iter::once(1usize).chain(1..degree).map(|x| x as f32).collect();
        
        // regularize cpd priors to penalize higher number of parents
        // log domain to avoid underflow
        let mut cpd_sum: f32 = 0.0;
        let mut cpd_regularized_sum: f32 = 0.0;
        for (i, exp) in exponent_array.iter().enumerate() {
            let cpd_0 = (1.0 - alpha).powi(*exp as i32) * (1.0 - beta);
            let cpd_1 = 1.0 - cpd_0;
            cpd_sum += cpd_0 + cpd_1;
            cpd_array.push([cpd_0, cpd_1]);

            let cpd_regularized_0 = (cpd_0.powi(*exp as i32) * (1.0 - beta)) / divide_array[i];
            let cpd_regularized_1 = 1.0 - cpd_regularized_0;
            cpd_regularized_sum += cpd_regularized_0 + cpd_regularized_1;
            cpd_array_regularized.push([cpd_regularized_0, cpd_regularized_1]);
        }

        // Normalize arrays (assuming normalize and avoid_underflow are implemented)
        Self::normalize_cpd(&mut cpd_array, cpd_sum, false);
        Self::normalize_cpd(&mut cpd_array_regularized, cpd_regularized_sum, true);
        
        // Create factor
        let initial_belief = if regularized { cpd_array_regularized } else { cpd_array };
        
        // Add factor to the node's attributes
        self.subtype = NodeType::Factor { initial_belief };
    }

    /// Normalizes CPD values in-place.
    ///
    /// # Arguments
    /// * `arr` - CPD array.
    /// * `sum` - Normalization constant.
    /// * `avoid_underflow` - If true, enforce minimum values.
    fn normalize_cpd(arr: &mut [[f32; 2]], sum: f32, avoid_underflow: bool) {
        for cpd in arr.iter_mut() {
            cpd[0] /= sum;
            cpd[1] /= sum;
    
            if avoid_underflow {
                if cpd[0] < 1e-30 {
                    cpd[0] = 1e-30;
                }
                if cpd[1] < 1e-30 {
                    cpd[1] = 1e-30;
                }
            }
        }
    }

    /// Parses a GraphML node from a node id and data map.
    ///
    /// # Arguments
    /// * `id` - New numeric node ID.
    /// * `name` - GraphML node id string.
    /// * `current_node_data` - Parsed `<data>` fields for the node.
    ///
    /// # Returns
    /// A new `Node` or error if subtype is unknown.
    pub fn parse_node_from_parts(
        id: usize,
        name: String,
        current_node_data: HashMap<String, String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let subtype: NodeType = match current_node_data.get("type").map(String::as_str) {
            Some("input") => {
                let belief_str = current_node_data
                    .get("belief")
                    .ok_or("belief not found while parsing input node")?;

                let beliefs: Vec<f32> = belief_str
                    .trim_matches(|c| c == '[' || c == ']')
                    .trim()
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()?;

                let belief: [f32; 2] = beliefs
                .as_slice()
                .try_into()
                .map_err(|_| "belief must contain exactly 2 values")?;

                NodeType::Variable {
                    output: false,
                    name: name.clone(),
                    initial_belief: belief,
                }
            }
            Some("output") => NodeType::Variable {
                output: true,
                name: name.clone(),
                initial_belief: [0.0, 0.0],
            },
            _ => {
                return Err("Node data has unknown type".into());
            }
        };

        Ok(Self {
            id: id as u32,
            incident_edges: Vec::new(),
            subtype,
        })
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a dummy factor node for unit tests.
    fn dummy_factor_node(id: usize) -> Node {
        Node::new(
            id,
            NodeType::Factor {
                initial_belief: Vec::new(),
            },
        )
    }

    /// Tests node creation and basic getters for variable nodes.
    #[test]
    fn test_new_and_getters() {
        let node = Node::new(1, NodeType::Variable { output: false, name: "node1".to_string(), initial_belief: [0.3, 0.7] });
        assert_eq!(node.get_id(), 1);
        assert!(!node.is_factor_node());
    }

    /// Verifies copy_with_id creates a new node with the updated ID.
    #[test]
    fn test_copy_with_id() {
        let node = Node::new(1, NodeType::Variable { output: false, name: "node1".to_string(), initial_belief: [0.0, 1.0] });
        let copy = node.copy_with_id(42);
        assert_eq!(copy.get_id(), 42);
    }

    /// Checks creation of convolution tree nodes.
    #[test]
    fn test_new_convolution_node() {
        let node = Node::new_convolution_node(10);
        assert_eq!(node.get_id(), 10);
    }

    /// Tests adding, retrieving, and setting incident edges on a node.
    #[test]
    fn test_incident_edges() {
        let mut node = Node::new(1, NodeType::Variable { output: false, name: "node1".to_string(), initial_belief: [0.1, 0.9] });
        node.add_incident_edge(5);
        assert_eq!(node.neighbors_count(), 1);
        assert_eq!(node.get_incident_edge(0), 5);
        let incident_edges: Vec<usize> = node.get_incident_edges().collect();
        assert_eq!(incident_edges, vec![5usize]);
        node.set_incident_edges(vec![7, 8].into_iter());
        let incident_edges: Vec<usize> = node.get_incident_edges().collect();
        assert_eq!(incident_edges, vec![7usize, 8usize]);
    }

    /// Verifies that node subtype updates and retrieval work properly.
    #[test]
    fn test_set_and_get_subtype() {
        let mut node = Node::new(1, NodeType::Factor { initial_belief: Vec::new() });
        node.set_subtype(NodeType::Variable { output: false, name: "node1".to_string(), initial_belief: [0.5, 0.5] });
        assert!(matches!(node.get_subtype(), NodeType::Variable { .. }));
    }

    /// Ensures fill_in_prior updates output variable priors correctly.
    #[test]
    fn test_fill_in_prior() {
        let mut node = Node::new(1, NodeType::Variable { output: true, name: "node1".to_string(), initial_belief: [0.0, 0.0] });
        node.fill_in_prior(0.8);
        if let NodeType::Variable { initial_belief, .. } = node.get_subtype() {
            assert!((initial_belief[0] - 0.2).abs() < 1e-5);
            assert!((initial_belief[1] - 0.8).abs() < 1e-5);
        } else {
            panic!("expected variable node");
        }
    }

    /// Checks that factor nodes receive a populated CPD when fill_in_factor is called.
    #[test]
    fn test_fill_in_factor() {
        let mut node = dummy_factor_node(2);
        node.add_incident_edge(0);
        node.fill_in_factor(1, 0.5, 0.1, false);
        if let NodeType::Factor { initial_belief, .. } = node.get_subtype() {
            assert!(!initial_belief.is_empty());
        } else {
            panic!("expected FactorNode");
        }
    }

    /// Verifies CPD normalization maintains probabilities and underflow protection.
    #[test]
    fn test_normalize_cpd() {
        let mut arr = vec![[2.0, 2.0], [1.0, 3.0]];
        Node::normalize_cpd(&mut arr, 8.0, true);
        for row in arr {
            assert!(row[0] >= 1e-30);
            assert!(row[1] >= 1e-30);
            assert!((row[0] + row[1]) <= 1.0 + 1e-9);
        }
    }
}
