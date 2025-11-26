use rustfft::{FftPlanner, num_complex::Complex, num_traits::Zero};
use crate::array_utils::normalize;

#[derive(Debug, Clone)]
struct CTNode {
    joint_above: Vec<Vec<f64>>,
    likelihood_below: Option<Vec<Vec<f64>>>,
}

impl CTNode {
    /// Creates a new CTNode with a given probability vector.
    /// 
    /// Normalizes the input vector so that it sums to 1.
    /// 
    /// # Arguments
    /// * `joint_above` - A vector of probabilities for this node.
    fn new(mut joint_above: Vec<Vec<f64>>) -> Self {
        for joint in &mut joint_above {
            normalize(joint);
        }

        Self {
            joint_above: joint_above,
            likelihood_below: None,
        }
    }

    /// Creates a count node by convolving the joint distributions of two child nodes.
    /// 
    /// # Arguments
    /// * `lhs` - Left child node.
    /// * `rhs` - Right child node.
    /// 
    /// # Returns
    /// A new CTNode representing the combined distribution.
    fn create_count_node(lhs: &CTNode, rhs: &CTNode) -> CTNode {
        let mut joint_above = Vec::with_capacity(lhs.joint_above.len());
        for (joint_lhs, joint_rhs) in lhs.joint_above.iter().zip(rhs.joint_above.iter()) {
            joint_above.push(fft_convolve(joint_lhs, joint_rhs));
        }

        let node = CTNode::new(joint_above);
        node
    }

    /// Computes the upward message by convolving the likelihood below with a sibling's distribution.
    /// 
    /// # Arguments
    /// * `answer_size` - Size of the output message.
    /// * `other_joint_vector` - The joint distribution of the sibling node.
    /// 
    /// # Returns
    /// A normalized probability vector representing the upward message.
    fn message_up(&self, answer_size: usize, other_joint_vector: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(other_joint_vector.len());
        let likelihood_below = self.likelihood_below.as_ref().expect("Likelihood below is None!");
        for (likelihood, joint_vector) in likelihood_below.iter().zip(other_joint_vector.iter()) {
            let starting_point = joint_vector.len() - 1;
            let new_likelihood = fft_convolve(
                &joint_vector.iter().rev().cloned().collect::<Vec<f64>>(),
                likelihood,
            );
            let mut new_likelihood = new_likelihood[starting_point..starting_point + answer_size].to_vec();
            normalize(&mut new_likelihood);
            result.push(new_likelihood)
        }

        result
    }

    /// Returns the likelihood messages from below for this node.
    /// 
    /// # Returns
    /// A vector of probabilities from the likelihood below.
    fn messages_up(&self) -> &Vec<Vec<f64>> {
        self.likelihood_below.as_ref().expect("Likelihood below is None!")
    }
}

#[derive(Debug)]
pub struct ConvolutionTree {
    n_to_shared_likelihoods: Vec<f64>,
    log_length: usize,
    all_layers: Vec<Vec<CTNode>>,
    variable_layer: Vec<CTNode>,
    n_variables: usize
}

impl ConvolutionTree {
    /// Constructs a ConvolutionTree given shared likelihoods and protein probability vectors.
    /// 
    /// # Arguments
    /// * `n_to_shared_likelihoods` - Shared likelihoods vector.
    /// * `proteins` - A vector of protein probability distributions.
    /// 
    /// # Returns
    /// A fully constructed ConvolutionTree with messages propagated backward.
    pub fn new(n_to_shared_likelihoods: &Vec<f64>, variables: &Vec<f64>, var_len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let n_variables = variables.len() / var_len;
        let log_length = (n_variables as f64).log2().ceil() as usize;
        let mut tree = ConvolutionTree {
            n_to_shared_likelihoods: n_to_shared_likelihoods.clone(),
            log_length,
            all_layers: Vec::new(), 
            variable_layer: Vec::new(),
            n_variables: n_variables
        };

        tree.build_first_layer(variables, var_len);
        tree.build_remaining_layers()?;
        tree.propagate_backward();

        Ok(tree)
    }

    /// Builds the first layer of the tree from protein probability vectors.
    /// Pads with dummy nodes to make the layer length a power of 2.
    /// 
    /// # Arguments
    /// * `proteins` - A vector of protein probability distributions.
    fn build_first_layer(&mut self, variables: &Vec<f64>, var_len: usize) {
        let mut layer = Vec::with_capacity(2usize.pow(self.log_length as u32));
        for variable in variables.chunks_exact(var_len) {
            let likelihood_counts: Vec<Vec<f64>> = variable.iter().map(|&v| vec![1.0 - v, v]).collect();
            layer.push(CTNode::new(likelihood_counts));
        }

        // Pad with dummy nodes to make the length a power of 2
        while layer.len() < 2usize.pow(self.log_length as u32) {
            layer.push(CTNode::new(vec![vec![1.0, 0.0]; var_len]));
        }

        self.all_layers.push(layer);
    }

    /// Builds all remaining layers of the tree by combining nodes into count nodes.
    /// Each layer is constructed by convolving pairs of nodes from the previous layer.
    fn build_remaining_layers(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..self.log_length {
            let most_recent_layer = self.all_layers.last().ok_or("last() called on an empty vector")?;
            let mut new_layer = Vec::new();

            for i in (0..most_recent_layer.len()).step_by(2) {
                let left = &most_recent_layer[i];
                let right = &most_recent_layer[i + 1];
                new_layer.push(CTNode::create_count_node(left, right));
            }

            self.all_layers.push(new_layer);
        }

        let likelihood_below: Vec<Vec<f64>> = self.n_to_shared_likelihoods.chunks(self.n_variables+1).map(|chunk| {
            let mut likelihood = chunk.to_vec();
            normalize(&mut likelihood);
            likelihood
        }).collect();
        self.all_layers.last_mut().ok_or("last() called on an empty vector")?[0].likelihood_below = Some(likelihood_below);

        Ok(())
    }

    /// Propagates likelihood messages from the root down to the protein nodes.
    /// Updates each node's `likelihood_below` using upward messages from parent nodes.
    fn propagate_backward(&mut self) {
        for l in (1..=self.log_length).rev() {
            for i in 0..self.all_layers[l].len() {
                
                let left_parent = &self.all_layers[l-1][2*i];
                let right_parent = &self.all_layers[l-1][2*i + 1];
                let node = &self.all_layers[l][i];

                let likelihood_below_left = Some(node.message_up(left_parent.joint_above[0].len(), &right_parent.joint_above));
                let likelihood_below_right = Some(node.message_up(right_parent.joint_above[0].len(), &left_parent.joint_above));

                self.all_layers[l-1][2*i].likelihood_below = likelihood_below_left;
                self.all_layers[l-1][2*i+1].likelihood_below = likelihood_below_right;
            }
        }

        self.variable_layer = self.all_layers[0].clone();
    }

    /// Retrieves the message to a variable node (protein).
    /// 
    /// # Arguments
    /// * `prot_idx` - Index of the protein node.
    /// 
    /// # Returns
    /// A probability vector representing the message to that protein.
    pub fn message_to_variable(&self, var_idx: usize) -> Vec<f64> {
        let mut message = self.variable_layer[var_idx].messages_up().iter().map(|l| l[1]).collect();
        normalize(&mut message);
        message
    }

    /// Retrieves the message to the shared likelihood node.
    /// 
    /// # Returns
    /// A probability vector representing the message to the shared likelihood.
    pub fn message_to_shared_likelihood(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>>{
        // Extract the required range
        let root = &self.all_layers.last().ok_or("last() called on an empty vector")?[0];
        let variable_joints = root.joint_above.iter().map(|joint| joint[..self.n_variables].to_vec());
        let shared_likelihood = variable_joints.flatten().collect();
        Ok(shared_likelihood)
    }
}


/// Performs FFT-based convolution of two probability vectors.
/// 
/// # Arguments
/// * `a` - First probability vector.
/// * `b` - Second probability vector.
/// 
/// # Returns
/// A new vector representing the convolution of `a` and `b`.
fn fft_convolve(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let len = a.len() + b.len() - 1;
    let fft_size = len.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut a_complex: Vec<Complex<f64>> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut b_complex: Vec<Complex<f64>> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    a_complex.resize(fft_size, Complex::zero());
    b_complex.resize(fft_size, Complex::zero());

    fft.process(&mut a_complex);
    fft.process(&mut b_complex);

    let mut result_complex: Vec<Complex<f64>> = a_complex.iter().zip(&b_complex).map(|(x, y)| x * y).collect();
    ifft.process(&mut result_complex);

    result_complex.iter().take(len).map(|c| c.re / fft_size as f64).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctnode_new_normalizes() {
        let node = CTNode::new(vec![2.0, 2.0]);
        assert!((node.joint_above[0] - 0.5).abs() < 1e-10);
        assert!((node.joint_above[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_create_count_node_convolution() {
        let lhs = CTNode::new(vec![1.0, 0.0]);
        let rhs = CTNode::new(vec![0.0, 1.0]);
        let node = CTNode::create_count_node(lhs, rhs);
        assert_eq!(node.joint_above.len(), 3);
        let sum: f64 = node.joint_above.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_message_up_and_messages_up() {
        let mut node = CTNode::new(vec![0.5, 0.5]);
        node.likelihood_below = Some(vec![0.5, 0.5]);
        let sibling_joint = vec![0.5, 0.5];
        let msg = node.message_up(2, &sibling_joint);
        let sum: f64 = msg.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        let msgs = node.messages_up();
        assert_eq!(msgs, vec![0.5, 0.5]);
    }

    #[test]
    fn test_convolution_tree_message_to_variable() {
        let shared = vec![0.5, 0.5];
        let proteins = vec![[1.0, 0.0], [0.0, 1.0]];
        let tree = ConvolutionTree::new(shared, proteins);
        assert!(tree.is_ok());
        let tree = tree.unwrap();

        let msg = tree.message_to_variable(0);
        assert!((msg.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolution_tree_message_to_shared_likelihood() {
        let shared = vec![0.5, 0.5];
        let proteins = vec![[1.0, 0.0], [0.0, 1.0]];
        let tree = ConvolutionTree::new(shared.clone(), proteins);
        assert!(tree.is_ok());
        let tree = tree.unwrap();

        let msg = tree.message_to_shared_likelihood();
        assert!(msg.is_ok());
        let msg = msg.unwrap();
        assert_eq!(msg.len(), tree.n_proteins + 1);
    }

    #[test]
    fn test_fft_convolve_basic() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let result = fft_convolve(&a, &b);
        let expected = vec![3.0, 10.0, 8.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-8);
        }
    }
}
