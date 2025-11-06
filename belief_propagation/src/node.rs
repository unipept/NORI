use std::collections::HashMap;
use minidom::Element;
use serde::Serialize;
use std::fmt::Write;


/// Defines the type of node in the factor graph with its initial beliefs.
#[derive(Debug, Serialize, Clone)]
pub enum NodeType {
    /// Variable node with prior probabilities.
    VariableNode { output: bool, name: String, initial_belief: [f64; 2] },
    /// Factor node with CPD.
    FactorNode { initial_belief: Vec<[f64; 2]> },
    /// Convolution tree node.
    ConvolutionTreeNode
}


/// Represents a node in the factor graph with its attributes and connections.
#[derive(Debug, Clone)]
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

    pub fn get_name(&self) -> Result<&str, Box<dyn std::error::Error>> {
        if let NodeType::VariableNode { name, .. } = self.get_subtype() {
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
        Self { id: id as u32, incident_edges: Vec::new(), subtype: NodeType::ConvolutionTreeNode }
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
        matches!(self.subtype, NodeType::FactorNode { .. })
    }

    pub fn is_output_node(&self) -> bool {
        matches!(self.subtype, NodeType::VariableNode { output: true , .. })
    }

    pub fn is_input_node(&self) -> bool {
        matches!(self.subtype, NodeType::VariableNode { output: false , .. })
    }

    /// Checks if the node is a convolution node.
    pub fn is_convolution_tree_node(&self) -> bool {
        matches!(self.subtype, NodeType::ConvolutionTreeNode { .. })
    }

    /// Updates prior belief for variable nodes.
    ///
    /// # Arguments
    /// * `prior` - Probability to assign as active.
    pub fn fill_in_prior(&mut self, prior: f64) {
        if let NodeType::VariableNode { output, name, .. } = &self.subtype {
            self.subtype = NodeType::VariableNode { output: *output, name: name.to_string(), initial_belief: [1.0 - prior, prior] };
        }
    }

    /// Initializes factor CPD with noisy-OR parameters.
    ///
    /// # Arguments
    /// * `alpha` - Peptide detection probability.
    /// * `beta` - Noise parameter.
    /// * `regularized` - Whether to apply parent-count regularization.
    pub fn fill_in_factor(&mut self, alpha: f64, beta: f64, regularized: bool) {
        if self.is_factor_node() {
            let degree: usize = self.neighbors_count();

            let mut cpd_array: Vec<[f64; 2]> = Vec::with_capacity(degree);
            let mut cpd_array_regularized = Vec::with_capacity(degree);
            let exponent_array: Vec<usize> = (0..degree).collect();
            let divide_array: Vec<f64> = std::iter::once(1usize).chain(1..degree).map(|x| x as f64).collect();
            
            // regularize cpd priors to penalize higher number of parents
            // log domain to avoid underflow
            let mut cpd_sum: f64 = 0.0;
            let mut cpd_regularized_sum: f64 = 0.0;
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
            self.subtype = NodeType::FactorNode { initial_belief };
        }
    }

    /// Normalizes CPD values in-place.
    ///
    /// # Arguments
    /// * `arr` - CPD array.
    /// * `sum` - Normalization constant.
    /// * `avoid_underflow` - If true, enforce minimum values.
    fn normalize_cpd(arr: &mut Vec<[f64; 2]>, sum: f64, avoid_underflow: bool) {
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

    /// Parses key-value data from a GraphML element.
    ///
    /// # Arguments
    /// * `data` - GraphML `<data>` element.
    ///
    /// # Returns
    /// A `(key, value)` pair.
    fn parse_data(data: &Element) -> Result<(String, String), Box<dyn std::error::Error>> {
        let key = data.attr("key").ok_or("key not found while parsing Node data")?.to_string();
        let value = data.text();
    
        Ok((key, value))
    }

    /// Parses a GraphML `<node>` element into a `Node`.
    ///
    /// # Arguments
    /// * `node` - GraphML element to parse.
    /// * `id` - Node ID.
    ///
    /// # Returns
    /// A new `Node` or error if subtype is unknown.
    pub fn parse_node(node: &Element, id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        // Process a node
        let name = node.attr("id").ok_or("id not found while parsing Node")?.to_string();
    
        // Initialize data for this node
        let mut current_node_data = HashMap::new();
        for data in node.children().filter(|d| d.name() == "data") {
            let (data_key, data_val) = Self::parse_data(data)?;
            current_node_data.insert(data_key, data_val);
        }
    
        let subtype: NodeType = match current_node_data.get("type").map(String::as_str) {
            Some("input") => {
                let belief_str = current_node_data.get("belief").ok_or("belief not found while parsing input node")?;

                let beliefs: Vec<f64> = belief_str
                    .trim_matches(|c| c == '[' || c == ']').trim()
                    .split(',').map(|s| s.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()?;

                let belief: [f64; 2] = [beliefs[0], beliefs[1]];

                NodeType::VariableNode { output: false, name: name.clone(), initial_belief: belief }
            },
            Some("output") => {
                NodeType::VariableNode { output: true, name: name.clone(), initial_belief: [0.0, 0.0] }
            },
            _ => {
                return Err("Node data has unknown type".into());
            }
        };

        Ok(Self { id: id as u32, incident_edges: Vec::new(), subtype })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use minidom::Element;

    fn dummy_factor_node(id: usize) -> Node {
        Node::new(
            id,
            NodeType::FactorNode {
                initial_belief: vec::new(),
            },
        )
    }

    #[test]
    fn test_new_and_getters() {
        let node = Node::new(1, NodeType::VariableNode { output: false, name: "node1", initial_belief: [0.3, 0.7] });
        assert_eq!(node.get_id(), 1);
        assert!(!node.is_factor_node());
    }

    #[test]
    fn test_copy_with_id() {
        let node = Node::new(1, NodeType::VariableNode { output: false, name: "node1", initial_belief: [0.0, 1.0] });
        let copy = node.copy_with_id(42);
        assert_eq!(copy.get_id(), 42);
    }

    #[test]
    fn test_new_convolution_node() {
        let node = Node::new_convolution_node(10, 3);
        assert_eq!(node.get_id(), 10);
    }

    #[test]
    fn test_incident_edges() {
        let mut node = Node::new(1, NodeType::VariableNode { output: false, name: "node1", initial_belief: [0.1, 0.9] });
        node.add_incident_edge(5);
        assert_eq!(node.neighbors_count(), 1);
        assert_eq!(node.get_incident_edge(0), 5);
        let incident_edges: Vec<usize> = node.get_incident_edges().collect();
        assert_eq!(incident_edges, vec![5usize]);
        node.set_incident_edges(vec![7, 8].into_iter());
        let incident_edges: Vec<usize> = node.get_incident_edges().collect();
        assert_eq!(incident_edges, vec![7usize, 8usize]);
    }

    #[test]
    fn test_set_and_get_subtype() {
        let mut node = Node::new(1, NodeType::FactorNode { initial_belief: Vec::new() });
        node.set_subtype(NodeType::VariableNode { output: false, name: "node1", initial_belief: [0.5, 0.5] });
        assert!(matches!(node.get_subtype(), NodeType::VariableNode { .. }));
    }

    #[test]
    fn test_fill_in_prior() {
        let mut node = Node::new(1, NodeType::VariableNode { output: false, name: "node1", initial_belief: [0.0, 0.0] });
        node.fill_in_prior(0.8);
        if let NodeType::VariableNode { initial_belief } = node.get_subtype() {
            assert!((*initial_belief[0] - 0.2).abs() < 1e-9);
            assert!((*initial_belief[1] - 0.8).abs() < 1e-9);
        } else {
            panic!("expected variable node");
        }
    }

    #[test]
    fn test_fill_in_factor() {
        let mut node = dummy_factor_node(2, 2);
        node.fill_in_factor(0.5, 0.1, false);
        if let NodeType::FactorNode { initial_belief, .. } = node.get_subtype() {
            assert!(!initial_belief.array.is_empty());
        } else {
            panic!("expected FactorNode");
        }
    }

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

    #[test]
    fn test_parse_data() {
        let xml: &str = r#"<data xmlns="ns" key="d0">0.5</data>"#;
        let elem: Element = xml.parse().unwrap();
        let node_data = Node::parse_data(&elem);
        assert!(node_data.is_ok());
        let (k, v) = node_data.unwrap();

        assert_eq!(k, "d0");
        assert_eq!(v, "0.5");
    }

    #[test]
    fn test_parse_node_peptide() {
        let xml: &str = r#"<node xmlns="ns" id="n1">
            <data key="d2">peptide</data>
            <data key="d0">0.1</data>
            <data key="d1">0.9</data>
        </node>"#;
        let elem: Element = xml.parse().unwrap();
        let node = Node::parse_node(&elem, 7).unwrap();
        assert_eq!(node.get_id(), 7);
    }
}
