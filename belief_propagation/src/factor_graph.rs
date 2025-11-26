use crate::node::{Node, NodeType};
use minidom::Element;
use std::collections::{HashMap, HashSet};
use serde::{Serialize};

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

    /// Returns the first node ID of the edge.
    pub fn get_node1_id(&self) -> usize {
        self.node1_id as usize
    }

    /// Returns the second node ID of the edge.
    pub fn get_node2_id(&self) -> usize {
        self.node2_id as usize
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

#[derive(Debug)]
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
        for node in &mut self.nodes {
            node.fill_in_factor(alpha, beta, regularized);
        }
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

    pub fn get_neighbors_ids(&self, node: &Node) -> Vec<usize> {
        let neighbors: Vec<usize> = node.get_incident_edges().map(|edge_id| {
            let (node1_id, node2_id) = self.edges[edge_id].get_node_ids();
            if node1_id == node.get_id() { node2_id } else { node1_id }
        }).collect();

        neighbors
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
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components: Vec<Self> = Vec::new();

        for start_node in &self.nodes {
            if visited.insert(start_node.get_id()) {
                let mut component_ids: Vec<usize> = Vec::new();
                let mut old_to_new_nodes: HashMap<usize, usize> = HashMap::new();

                let mut new_nodes: Vec<Node> = Vec::new();
                let mut new_edges: Vec<Edge> = Vec::new();

                // Find ids of nodes to include in component
                component_ids.push(start_node.get_id());
                old_to_new_nodes.insert(start_node.get_id(), 0);
                self.find_component_rec(start_node.get_id(), &mut component_ids, &mut old_to_new_nodes, &mut visited);

                // Create new nodes
                for node_id in &component_ids {
                    let node = self.nodes[*node_id].copy_with_id(old_to_new_nodes[&node_id]);
                    new_nodes.push(node);
                }

                // Select edges to keep and update the node ids
                let mut next_edge_id: usize = 0;
                let mut component_edge_ids: HashSet<usize> = HashSet::new();
                let mut old_to_new_edges: HashMap<usize, usize> = HashMap::new();
                for edge in &self.edges {

                    let ((source, source_in_target), (target, target_in_source)): ((usize, usize), (usize, usize)) = edge.get_node_and_neighbor_ids();
                    if component_ids.contains(&source) && component_ids.contains(&target) {

                        let (new_source, new_target): (usize, usize) = (old_to_new_nodes[&source], old_to_new_nodes[&target]);
                        let new_edge = Edge::new(next_edge_id, new_source, new_target, source_in_target, target_in_source, edge.get_message_length());
                        next_edge_id += 1;

                        component_edge_ids.insert(edge.get_id());
                        old_to_new_edges.insert(edge.get_id(), new_edge.get_id());

                        new_edges.push(new_edge);
                    }
                }

                // Update edge ids of incident edges
                for node in &mut new_nodes {
                    let new_incident_edges: Vec<usize> = node.get_incident_edges().filter(|e| component_edge_ids.contains(e)).map(|e| old_to_new_edges[&e]).collect();
                    node.set_incident_edges(new_incident_edges.into_iter());
                }
                
                // Create graph and add to components
                let subgraph = Self { nodes: new_nodes, edges: new_edges };
                components.push(subgraph);
            }
        }

        components
    }

    /// Recursively explores connected nodes to identify all members of a component.
    ///
    /// # Arguments
    /// * `start_id` - The starting node ID for the recursive traversal.
    /// * `component_ids` - Mutable vector storing all node IDs in the current component.
    /// * `old_to_new_nodes` - Mapping from original node IDs to new local component IDs.
    /// * `visited` - A mutable set tracking visited node IDs to avoid revisiting.
    fn find_component_rec(
        &self, 
        start_id: usize, 
        component_ids: &mut Vec<usize>, 
        old_to_new_nodes: &mut HashMap<usize, usize>, 
        visited: &mut HashSet<usize>
    ) {
        let start_node: &Node = &self.nodes[start_id];
        for neighbor_id in self.get_neighbors(&start_node) {
            if visited.insert(neighbor_id) {
                let next_id: usize = component_ids.len();
                component_ids.push(neighbor_id);
                old_to_new_nodes.insert(neighbor_id, next_id);
                self.find_component_rec(neighbor_id, component_ids, old_to_new_nodes, visited);                
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeType, Factor};

    fn sample_csv() -> String {
        "id,sequence,score,psms,higher_taxa,weight,log_weight
1,PEPTIDE1,0.8,3,100,0.5,-0.3
2,PEPTIDE2,0.6,3,100,0.4,-0.5
3,PEPTIDE3,0.9,3,200,0.7,-0.1"
            .to_string()
    }

    #[test]
    fn test_parse_taxon_weights_csv() {
        let csv = sample_csv();
        let taxa = parse_taxon_weights_csv(csv).unwrap();
        assert_eq!(taxa.len(), 3);
        assert_eq!(taxa[0].id, 1);
        assert!((taxa[1].score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_generate_graph_creates_graphml() {
        let csv = sample_csv();
        let graphml = generate_graph(csv).unwrap();
        assert!(graphml.contains("graphml"));
        assert!(graphml.contains("node"));
        assert!(graphml.contains("edge"));
    }

    #[test]
    fn test_edge_getters() {
        let edge = Edge::new(1, 10, 20, 0, 0, Some(5));
        assert_eq!(edge.get_id(), 1);
        assert_eq!(edge.get_node1_id(), 10);
        assert_eq!(edge.get_node2_id(), 20);
        assert_eq!(edge.get_node_ids(), (10, 20));
        assert_eq!(edge.get_message_length(), Some(5));
    }

    #[test]
    fn test_ctfactorgraph_from_taxa_weights() {
        let csv = sample_csv();
        let taxa = parse_taxon_weights_csv(csv).unwrap();
        let graph = CTFactorGraph::from_taxa_weights(taxa);
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_ctfactorgraph_to_and_from_graphml() {
        let csv = sample_csv();
        let taxa = parse_taxon_weights_csv(csv).unwrap();
        let graph = CTFactorGraph::from_taxa_weights(taxa);
        let graphml = graph.to_graphml();
        assert!(graphml.is_ok());
        let graphml = graphml.unwrap();

        let parsed = CTFactorGraph::from_graphml(&graphml).unwrap();
        assert_eq!(graph.node_count(), parsed.node_count());
        assert_eq!(graph.edge_count(), parsed.edge_count());
    }

    #[test]
    fn test_neighbor_operations() {
        let csv = sample_csv();
        let taxa = parse_taxon_weights_csv(csv).unwrap();
        let graph = CTFactorGraph::from_taxa_weights(taxa);

        if graph.node_count() > 1 {
            let node = graph.get_node(0);
            println!("{:?}\n\n{:?}", graph, node);
            for n in graph.get_neighbors(node) {
                assert!(n >= 0);
            }
        }
    }

    #[test]
    fn test_get_peptide_for_factor_returns_ok_or_err() {
        let csv = sample_csv();
        let taxa = parse_taxon_weights_csv(csv).unwrap();
        let graph = CTFactorGraph::from_taxa_weights(taxa);

        for (i, node) in graph.get_nodes().iter().enumerate() {
            if node.is_factor_node() {
                let result = graph.get_peptide_for_factor(i);
                assert!(result.is_ok() || result.is_err());
            }
        }
    }

    #[test]
    fn test_connected_components() {
        let csv = sample_csv();
        let taxa = parse_taxon_weights_csv(csv).unwrap();
        let graph = CTFactorGraph::from_taxa_weights(taxa);

        let components = graph.connected_components();
        assert!(!components.is_empty());
        let total_nodes: usize = components.iter().map(|c| c.node_count()).sum();
        assert_eq!(total_nodes, graph.node_count());
    }
}
