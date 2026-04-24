# Belief Propagation

A Rust library for performing zero-lookahead belief propagation on bipartite graphs with noisy-OR  models, optimized for large-scale biological data analysis.

## Overview

This repository contains:

- **`belief_propagation/`**: The core Rust library implementing belief propagation algorithms.
- **`benchmark_code/`**: Example code for benchmarking the library's performance.
- **`input_data/`**: Sample GraphML files for testing and demonstration.

## Features

- Zero-lookahead belief propagation for factor graphs
- Convolution tree optimization for high-degree nodes
- Support for noisy-OR factor tables and prior beliefs

## Installation

Ensure you have Rust installed. Clone the repository and build the library:

```bash
git clone <repository-url>
cd belief-propagation
cargo build --release
```

## Usage

### Loading and Running Belief Propagation

```rust
use belief_propagation::{load_factor_graph, zero_lookahead_bp_from_graph};

let graphs = load_factor_graph(graphml_string)?;
let result = zero_lookahead_bp_from_graph(&mut graphs, alpha, beta, regularized, prior, max_iter, tolerance)?;
```

#### Parameters for `zero_lookahead_bp_from_graph`

- `graphs`: Mutable reference to a vector of `CTFactorGraph` objects (loaded via `load_factor_graph`).
- `alpha`: `f32` - Noisy-OR alpha parameter (probability of correct detection, e.g., 0.9).
- `beta`: `f32` - Noisy-OR beta parameter (noise level, e.g., 0.1).
- `regularized`: `bool` - If `true`, regularizes factor tables to penalize high-degree nodes.
- `prior`: `f32` - Prior probability for output nodes (e.g., 0.5 for uniform).
- `max_iter`: `Option<u32>` - Maximum iterations (default 10,000 if `None`).
- `tolerance`: `Option<f32>` - Convergence tolerance for messages (default 0.006 if `None`).

The function returns a CSV string with columns: `[node_name, belief_array]`.

## Input Format

Graphs are provided as GraphML strings representing a bipartite graph with input and output nodes, and edges indicating noisy-OR relationships.

