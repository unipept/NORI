use crate::node::{Node, NodeType};
use minidom::{Element};
use std::collections::HashMap;
use serde::Serialize;


/// Represents an edge in a factor graph connecting two nodes.
#[derive(Debug, Serialize, Clone)]
pub struct Edge {
    id: u32,
    node1_id: u32,
    node2_id: u32,
    node1_in_node2_id: u32,
    node2_in_node1_id: u32,
    message_length: Option<u32>
}


impl Edge {

    /// Creates a new edge connecting two nodes in a factor graph.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the edge.
    /// * `node1_id` - ID of the first node connected by this edge.
    /// * `node2_id` - ID of the second node connected by this edge.
    /// * `message_length` - Optional message length associated with the edge. Can be `None` if not applicable.
    ///
    /// # Returns
    /// An `Edge` instance representing a connection between the two specified nodes.
    pub fn new(id: usize, node1_id: usize, node2_id: usize, node1_in_node2_id: usize, node2_in_node1_id: usize, message_length: Option<usize>) -> Edge {
        Edge { 
            id: id as u32, 
            node1_id: node1_id as u32, 
            node2_id: node2_id as u32, 
            node1_in_node2_id: node1_in_node2_id as u32, 
            node2_in_node1_id: node2_in_node1_id as u32, 
            message_length: message_length.map(|x| x as u32) 
        }
    }

    /// Sets the ID of the current node within the context of node2.
    ///
    /// # Arguments
    /// * `id` - The index of this node within the neighbor list of node2.
    pub fn set_node1_in_node2_id(&mut self, id: usize) {
        self.node1_in_node2_id = id as u32;
    }

    /// Sets the ID of the second node within the context of node1.
    ///
    /// # Arguments
    /// * `id` - The index of node2 within the neighbor list of node1.
    pub fn set_node2_in_node1_id(&mut self, id: usize) {
        self.node2_in_node1_id = id as u32;
    }

    /// Returns the ID of the edge.
    pub fn get_id(&self) -> usize {
        self.id as usize
    }

    /// Returns a tuple of the two node IDs of the edge.
    pub fn get_node_ids(&self) -> (usize, usize) {
        (self.node1_id as usize, self.node2_id as usize)
    }

    /// Returns both node IDs and their corresponding neighbor indices.
    pub fn get_node_and_neighbor_ids(&self) -> ((usize, usize), (usize, usize)) {
        ((self.node1_id as usize, self.node1_in_node2_id as usize), (self.node2_id as usize, self.node2_in_node1_id as usize))
    }

    /// Returns the message length associated with the edge.
    pub fn get_message_length(&self) -> Option<usize> {
        self.message_length.map(|x| x as usize)
    }

    /// Creates a copy of the current edge with a new unique edge ID.
    ///
    /// # Arguments
    /// * `new_id` - The new edge ID to assign.
    pub fn copy_with_id(&self, new_id: usize) -> Self {
        let mut copy: Edge = self.clone();
        copy.id = new_id as u32;
        copy
    }
}

#[derive(Debug, Clone)]
pub struct CTFactorGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl CTFactorGraph {

    /// Creates a new factor graph from a list of nodes and edges.
    ///
    /// # Arguments
    /// * `nodes` - A vector of `Node` instances representing all nodes in the graph.
    /// * `edges` - A vector of `Edge` instances representing all edges connecting the nodes.
    ///
    /// # Returns
    /// A `CTFactorGraph` instance containing the provided nodes and edges.
    pub fn new(nodes: Vec<Node>, edges: Vec<Edge>) -> CTFactorGraph {
        CTFactorGraph { nodes, edges }
    }

    /// Returns a reference to the node with the given ID.
    ///
    /// # Arguments
    /// * `node_id` - ID of the node to retrieve.
    ///
    /// # Returns
    /// A reference to the `Node` corresponding to `node_id`.
    pub fn get_node(&self, node_id: usize) -> &Node {
        &self.nodes[node_id]
    }

    /// Returns a reference to the edge with the given ID.
    ///
    /// # Arguments
    /// * `edge_id` - ID of the edge to retrieve.
    ///
    /// # Returns
    /// A reference to the `Edge` corresponding to `edge_id`.
    pub fn get_edge(&self, edge_id: usize) -> &Edge {
        &self.edges[edge_id]
    }

    /// Returns the total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns a reference to all nodes in the graph.
    pub fn get_nodes(&self) -> &Vec<Node> {
        &self.nodes
    }

    /// Returns a reference to all edges in the graph.
    pub fn get_edges(&self) -> &Vec<Edge> {
        &self.edges
    }

     /// Parses an XML `<edge>` element and extracts its source and target node names.
    ///
    /// # Arguments
    /// * `edge` - A reference to a `minidom::Element` representing an edge in GraphML.
    ///
    /// # Returns
    /// Returns a tuple `(source, target)` representing the IDs of connected nodes.
    fn parse_edge(edge: &Element) -> Result<(String, String), Box<dyn std::error::Error>> {
        let source: String = edge.attr("source").ok_or("Source attribute does not exist in Edge")?.to_string();
        let target: String = edge.attr("target").ok_or("Target attribute does not exist in Edge")?.to_string();
    
        Ok((source, target))
    }
    
    /// Constructs a `CTFactorGraph` from a GraphML string.
    ///
    /// # Arguments
    /// * `graphml_str` - A string containing a GraphML representation of a graph.
    ///
    /// # Returns
    /// A `Result` containing the constructed `CTFactorGraph` if successful.
    ///
    /// # Errors
    /// Returns an error if parsing the GraphML fails or if nodes/edges cannot be created correctly.
    pub fn from_graphml(graphml_str: &str) -> Result<CTFactorGraph, Box<dyn std::error::Error>> {
        let root: Element = graphml_str.parse()?;

        let node_count = root.children().filter(|n| n.name() == "graph").map(|g| g.children().filter(|n| n.name() == "node").count()).sum();
        let mut nodes: Vec<Node> = Vec::with_capacity(node_count);
        let edge_count = root.children().filter(|n| n.name() == "graph").map(|g| g.children().filter(|n| n.name() == "edge").count()).sum();
        let mut edges: Vec<Edge> = Vec::with_capacity(edge_count);
        let mut node_map: HashMap<String, usize> = HashMap::new();
        
        let mut next_node_id = 0;
        let mut next_edge_id = 0;
        for graph_xml in root.children().filter(|n| n.name() == "graph") {
            for node_xml in graph_xml.children().filter(|n| n.name() == "node") {
                let node: Node = Node::parse_node(node_xml, next_node_id)?;
                let node_name: String = node.get_name()?.to_string();
                node_map.insert(node_name, next_node_id);
                next_node_id += 1;

                nodes.push(node);
            }
    
            for edge_xml in graph_xml.children().filter(|n| n.name() == "edge") {
                let (source, target) = Self::parse_edge(edge_xml)?;
    
                let node1_id: usize = *node_map.get(&source).ok_or("Source node of edge not present in graph")?;
                let node2_id: usize = *node_map.get(&target).ok_or("Target node of edge not present in graph")?;
                let node1: &Node = &nodes[node1_id];
                let node2: &Node = &nodes[node2_id];
                let edge = Edge::new(next_edge_id, node1_id, node2_id, node2.neighbors_count(), node1.neighbors_count(), None);
                next_edge_id += 1;
    
                let node1: &mut Node = &mut nodes[node1_id];
                node1.add_incident_edge(edge.get_id());
                let node2: &mut Node = &mut nodes[node2_id];
                node2.add_incident_edge(edge.get_id());
                edges.push(edge);
            }
        }
        
        let mut graph = CTFactorGraph { nodes, edges };
        graph.add_factor_nodes();
    
        Ok( graph )
    }

    pub fn add_factor_nodes(&mut self) {
        let mut next_node_id = self.node_count();
        let mut next_edge_id = self.edge_count();
        let mut new_nodes = Vec::new();
        for node in &mut self.nodes {
            if node.is_input_node() {
                let mut new_variable_node = Node::new(next_node_id, node.get_subtype().clone());
                next_node_id += 1;
                node.set_subtype(NodeType::FactorNode { initial_belief: Vec::new() });
                new_nodes.push(new_variable_node.clone());

                let edge = Edge::new(next_edge_id, new_variable_node.get_id(), node.get_id(), node.neighbors_count(), 0, None);
                self.edges.push(edge);
                node.add_incident_edge(next_edge_id);
                new_variable_node.add_incident_edge(next_edge_id);
                next_edge_id += 1;
            }
        }

        for node in new_nodes { 
            self.nodes.push(node);
        }
    }

    /// Fills all nodes with a prior probability.
    ///
    /// # Arguments
    /// * `prior` - The prior probability to assign to each node.
    pub fn fill_in_priors(&mut self, prior: f64) {
        for node in &mut self.nodes {
            node.fill_in_prior(prior);
        }
    }

    /// Fills all factor nodes with factor probabilities using alpha/beta parameters.
    ///
    /// # Arguments
    /// * `alpha` - Alpha parameter for factor probability.
    /// * `beta` - Beta parameter for factor probability.
    /// * `regularized` - Whether to apply regularization.
    pub fn fill_in_factors(&mut self, alpha: f64, beta: f64, regularized: bool) {
        for i in 0..self.nodes.len() {
            if self.nodes[i].is_factor_node() {
                let mut degree = self.nodes[i].neighbors_count();
                let neighbor: &Node = &self.nodes[self.get_neighbor_node_and_neighbor_id(&self.nodes[i], 0).0];
                if neighbor.is_convolution_tree_node() {
                    degree = neighbor.neighbors_count();
                }
                self.nodes[i].fill_in_factor(degree, alpha, beta, regularized);
            }
        }
    }

    /// Returns the IDs of neighbors of a node, given its node ID.
    ///
    /// # Arguments
    /// * `node_id` - The node ID for which neighbors are requested.
    ///
    /// # Returns
    /// A vector of node IDs representing neighbors.
    pub fn get_neighbors_from_id(&self, node_id: usize) -> impl Iterator<Item = usize> + use<'_> {
        self.get_neighbors(self.get_node(node_id))
    }

    /// Returns the IDs of neighbors for a given node.
    ///
    /// # Arguments
    /// * `node` - Reference to the `Node` whose neighbors are requested.
    ///
    /// # Returns
    /// A Iterator over node IDs representing neighbors.
    pub fn get_neighbors<'a>(&'a self, node: &'a Node) -> impl Iterator<Item = usize> + 'a {
        node.get_incident_edges().map(|edge_id| {
            let (node1_id, node2_id) = self.edges[edge_id].get_node_ids();
            if node1_id == node.get_id() { node2_id } else { node1_id }
        })
    }

    /// Returns the node ID of a neighbor given a node and its neighbor ID.
    ///
    /// # Arguments
    /// * `node` - Reference to the node.
    /// * `neighbor_id` - Index of the neighbor within the nodes neighbors.
    ///
    /// # Returns
    /// Node ID of the neighbor.
    pub fn get_neighbor_node_id(&self, node: &Node, neighbor_id: usize) -> usize {
        let (node1_id, node2_id) = self.edges[node.get_incident_edge(neighbor_id)].get_node_ids();
        if node1_id == node.get_id() { node2_id } else { node1_id }
    }

    /// Returns both the neighboring node ID and the neighbor index of that node.
    ///
    /// # Arguments
    /// * `node` - Reference to the node whose neighbor is queried.
    /// * `neighbor_id` - The index of the neighbor within the node’s adjacency list.
    ///
    /// # Returns
    /// A tuple `(neighbor_node_id, neighbor_index_in_neighbor)` representing the relationship.
    pub fn get_neighbor_node_and_neighbor_id(&self, node: &Node, neighbor_id: usize) -> (usize, usize) {
        let ((node1_id, node1_in_node2_id), (node2_id, node2_in_node1_id)) = self.edges[node.get_incident_edge(neighbor_id)].get_node_and_neighbor_ids();
        if node1_id == node.get_id() { (node2_id, node1_in_node2_id) } else { (node1_id, node2_in_node1_id) }
    }

    /// Adds convolution tree nodes to the graph, creating edges appropriately.
    pub fn add_ct_nodes(&mut self) {
        // When creating the CTGraph and not just reading from a previously saved graph format, use this function to add the CT nodes
        let ct_node_count = self.nodes.iter().filter(|n| n.is_factor_node() && n.neighbors_count() > 2).count();
        let mut new_nodes: Vec<Node> = Vec::with_capacity(&self.nodes.len() + ct_node_count);
        new_nodes.extend_from_slice(&self.nodes);
        let mut new_edges: Vec<Edge> = Vec::with_capacity(&self.edges.len() + ct_node_count);

        // Add nodes and keep track of edges to add/remove
        let mut next_edge_id: usize = 0;
        let mut next_node_id: usize = self.nodes.len();
        for node in &self.nodes {
            if node.is_factor_node() {
                if node.neighbors_count() > 2 {
                
                    let new_node_id = next_node_id;
                    let new_node = Node::new_convolution_node(new_node_id);
                    next_node_id += 1;
                    new_nodes.push(new_node);

                    // Create edge Factor CTree, set node_in_node_id's to 0, we will set them correctly later
                    let edge = Edge::new(next_edge_id, new_node_id, node.get_id(), 0, 0, Some(node.neighbors_count()));
                    next_edge_id += 1;
                    new_edges.push(edge);

                    for (i, edge_id) in node.get_incident_edges().enumerate() {
                        let neighbor_id = self.get_neighbor_node_id(node, i);
                        let neighbor: &Node = self.get_node(neighbor_id);
                        if neighbor.is_output_node() {
                            // Create edge CTree - variable, set node_in_node_id's to 0, we will set them correctly later
                            let edge = Edge::new(next_edge_id, new_node_id, neighbor_id, 0, 0, None);
                            next_edge_id += 1;
                            new_edges.push(edge);
                        } else {
                            // Add Factor - Peptide node
                            new_edges.push(self.get_edge(edge_id).copy_with_id(next_edge_id));
                            next_edge_id += 1;
                        }
                    }
                } else {
                    for edge_id in node.get_incident_edges() {
                        new_edges.push(self.get_edge(edge_id).copy_with_id(next_edge_id));
                        next_edge_id += 1;
                    }
                }
                
            }
        }

        // Clear the incident edges of each node, and refill in the next step
        for node in &mut new_nodes {
            node.set_incident_edges(Vec::new().into_iter());
        }
        
        for edge in &mut new_edges {
            let (node1_id, node2_id) = edge.get_node_ids();
            edge.set_node1_in_node2_id(new_nodes[node2_id].neighbors_count());
            edge.set_node2_in_node1_id(new_nodes[node1_id].neighbors_count());
            new_nodes[node1_id].add_incident_edge(edge.get_id());
            new_nodes[node2_id].add_incident_edge(edge.get_id());
        }

        self.nodes = new_nodes;
        self.edges = new_edges;
    }

    /// Returns a vector of connected components in the graph as separate `CTFactorGraph`s.
    ///
    /// # Returns
    /// A vector of `CTFactorGraph` instances, one per connected component.
    pub fn connected_components(&self) -> Vec<Self> {
        let mut visited: Vec<bool> = vec![false; self.nodes.len()];
        let mut visited_edges: Vec<bool> = vec![false; self.edges.len()];
        let mut components: Vec<Self> = Vec::new();
        for start_node in &self.nodes {
            if ! visited[start_node.get_id()] {
                visited[start_node.get_id()] = true;
                let mut old_to_new_nodes: HashMap<usize, usize> = HashMap::new();

                let mut new_nodes: Vec<Node> = vec![start_node.copy_with_id(0)];
                let mut new_edges: Vec<Edge> = Vec::new();

                // Find ids of nodes to include in component
                old_to_new_nodes.insert(start_node.get_id(), 0);
                let mut stack = vec![start_node.get_id()];
                let mut new_local_id = 1;
                while let Some(node_id) = stack.pop() {
                    for neighbor_id in self.get_neighbors(&self.nodes[node_id]) {
                        if ! visited[neighbor_id] {
                            visited[neighbor_id] = true;
                            // assign a new index in the component
                            old_to_new_nodes.insert(neighbor_id, new_local_id);
                            new_nodes.push(self.nodes[neighbor_id].copy_with_id(new_local_id));
                            new_local_id += 1;

                            stack.push(neighbor_id);
                        }
                    }
                }

                // Select edges to keep and update the node ids
                let mut next_edge_id: usize = 0;
                let mut old_to_new_edges: HashMap<usize, usize> = HashMap::new();
                for node in &new_nodes {
                    for edge_id in node.get_incident_edges() {
                        if ! visited_edges[edge_id] {
                            visited_edges[edge_id] = true;
                            let edge: &Edge = self.get_edge(edge_id);
                            let ((source, source_in_target), (target, target_in_source)): ((usize, usize), (usize, usize)) = edge.get_node_and_neighbor_ids();

                            let (new_source, new_target): (usize, usize) = (old_to_new_nodes[&source], old_to_new_nodes[&target]);
                            let new_edge = Edge::new(next_edge_id, new_source, new_target, source_in_target, target_in_source, edge.get_message_length());
                            next_edge_id += 1;

                            old_to_new_edges.insert(edge_id, new_edge.get_id());

                            new_edges.push(new_edge);
                        }
                    }
                }

                // Update edge ids of incident edges
                for node in &mut new_nodes {
                    let new_incident_edges: Vec<usize> = node.get_incident_edges().map(|e| old_to_new_edges[&e]).collect();
                    node.set_incident_edges(new_incident_edges.into_iter());
                }
                
                // Create graph and add to components
                let subgraph = Self { nodes: new_nodes, edges: new_edges };
                components.push(subgraph);
            }
        }
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeType};

    /// Helper: build a simple graph with two variable nodes and one edge
    fn simple_graph() -> CTFactorGraph {
        let n0 = Node::new(
            0,
            NodeType::VariableNode {
                output: false,
                name: "A".into(),
                initial_belief: [0.4, 0.6],
            },
        );

        let n1 = Node::new(
            1,
            NodeType::VariableNode {
                output: true,
                name: "B".into(),
                initial_belief: [0.0, 0.0],
            },
        );

        let e0 = Edge::new(0, 0, 1, 0, 0, None);

        let mut graph = CTFactorGraph::new(vec![n0, n1], vec![e0]);
        graph.nodes[0].add_incident_edge(0);
        graph.nodes[1].add_incident_edge(0);
        graph
    }

    /* ----------------------------- Edge tests ----------------------------- */

    #[test]
    fn edge_new_and_getters() {
        let e = Edge::new(3, 1, 2, 4, 5, Some(10));

        assert_eq!(e.get_id(), 3);
        assert_eq!(e.get_node_ids(), (1, 2));
        assert_eq!(e.get_message_length(), Some(10));
    }

    #[test]
    fn edge_setters() {
        let mut e = Edge::new(0, 0, 1, 0, 0, None);

        e.set_node1_in_node2_id(7);
        e.set_node2_in_node1_id(9);

        let ((_, n1_in_n2), (_, n2_in_n1)) = e.get_node_and_neighbor_ids();
        assert_eq!(n1_in_n2, 7);
        assert_eq!(n2_in_n1, 9);
    }

    #[test]
    fn edge_copy_with_id() {
        let e = Edge::new(1, 2, 3, 4, 5, Some(6));
        let copy = e.copy_with_id(10);

        assert_eq!(copy.get_id(), 10);
        assert_eq!(copy.get_node_ids(), (2, 3));
        assert_eq!(copy.get_message_length(), Some(6));
    }

    /* -------------------------- Graph basic tests -------------------------- */

    #[test]
    fn graph_new_and_counts() {
        let g = simple_graph();

        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn graph_getters() {
        let g = simple_graph();

        assert_eq!(g.get_node(0).get_id(), 0);
        assert_eq!(g.get_edge(0).get_id(), 0);
        assert_eq!(g.get_nodes().len(), 2);
        assert_eq!(g.get_edges().len(), 1);
    }

    /* ---------------------------- GraphML tests ---------------------------- */

    #[test]
    fn from_graphml_parses_graph() {
        let graphml = r#"<?xml version='1.0' encoding='utf-8'?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
          <graph id="G" edgedefault="undirected">
            <node id="X">
              <data key="type">input</data>
              <data key="belief">[0.2, 0.8]</data>
            </node>
            <node id="Y">
              <data key="type">output</data>
            </node>
            <edge source="X" target="Y"/>
          </graph>
        </graphml>
        "#;

        let g = CTFactorGraph::from_graphml(graphml).unwrap();

        assert_eq!(g.node_count(), 3);
        assert!(g.nodes.iter().any(|n| n.is_factor_node()));
    }

    /* ------------------------- Factor-node creation ------------------------ */

    #[test]
    fn add_factor_nodes_creates_factor_and_edge() {
        let mut g = simple_graph();
        g.add_factor_nodes();

        assert!(g.nodes.iter().any(|n| n.is_factor_node()));
        assert!(g.edge_count() > 1);
    }

    /* --------------------------- Prior filling ----------------------------- */

    #[test]
    fn fill_in_priors_sets_output_nodes() {
        let mut g = simple_graph();
        g.fill_in_priors(0.7);

        for n in g.nodes.iter() {
            if let NodeType::VariableNode { output: true, initial_belief, .. } = n.get_subtype() {
                assert!((initial_belief[1] - 0.7).abs() < 1e-9);
            }
        }
    }

    /* -------------------------- Factor CPD filling ------------------------- */

    #[test]
    fn fill_in_factors_initializes_factor_nodes() {
        let mut g = simple_graph();
        g.add_factor_nodes();
        g.fill_in_factors(0.9, 0.1, false);

        for n in g.nodes.iter() {
            if let NodeType::FactorNode { initial_belief } = n.get_subtype() {
                assert!(!initial_belief.is_empty());
            }
        }
    }

    /* -------------------------- Neighbor queries ---------------------------- */

    #[test]
    fn get_neighbors_from_id() {
        let g = simple_graph();
        let neighbors: Vec<_> = g.get_neighbors_from_id(0).collect();

        assert_eq!(neighbors, vec![1]);
    }

    #[test]
    fn get_neighbor_node_and_neighbor_id() {
        let g = simple_graph();
        let node = g.get_node(0);
        let (neighbor, neighbor_index) = g.get_neighbor_node_and_neighbor_id(node, 0);

        assert_eq!(neighbor, 1);
        assert_eq!(neighbor_index, 0);
    }

    /* ---------------------- Convolution tree expansion --------------------- */

    #[test]
    fn add_ct_nodes_adds_convolution_nodes() {
        let graphml = r#"<?xml version='1.0' encoding='utf-8'?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
          <graph id="G" edgedefault="undirected">
            <node id="A">
              <data key="type">input</data>
              <data key="belief">[0.5,0.5]</data>
            </node>
            <node id="B">
              <data key="type">output</data>
            </node>
            <node id="C">
              <data key="type">output</data>
            </node>
            <edge source="A" target="B"/>
            <edge source="A" target="C"/>
          </graph>
        </graphml>
        "#;

        let mut g = CTFactorGraph::from_graphml(graphml).unwrap();
        g.add_ct_nodes();

        assert!(g.nodes.iter().any(|n| n.is_convolution_tree_node()));
    }

    /* ------------------------ Connected components ------------------------- */

    #[test]
    fn connected_components_splits_graph() {
        let n0 = Node::new(
            0,
            NodeType::VariableNode {
                output: false,
                name: "A".into(),
                initial_belief: [0.5, 0.5],
            },
        );
        let n1 = Node::new(
            1,
            NodeType::VariableNode {
                output: false,
                name: "B".into(),
                initial_belief: [0.5, 0.5],
            },
        );

        let g = CTFactorGraph::new(vec![n0, n1], vec![]);
        let components = g.connected_components();

        assert_eq!(components.len(), 2);
        assert_eq!(components[0].node_count(), 1);
        assert_eq!(components[1].node_count(), 1);
    }
}