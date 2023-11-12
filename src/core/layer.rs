use std::collections::HashMap;

use rand::Rng;

use nalgebra::DMatrix;

pub struct Layer {
    activation: fn(f64) -> f64,
    activation_derivative: fn(f64) -> f64,
    biases: DMatrix<f64>,
    deltas: DMatrix<f64>,
    errors: DMatrix<f64>,
    last_activated_output: DMatrix<f64>,
    last_raw_output: DMatrix<f64>,
    optimizer_params: HashMap<String, DMatrix<f64>>,
    weights: DMatrix<f64>,
}

impl Layer {
    pub fn new(
        activation: fn(f64) -> f64,
        activation_derivative: fn(f64) -> f64,
        input_dim: usize,
        neurons: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let weights: Vec<f64> = (0..input_dim * neurons)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let biases: Vec<f64> = (0..neurons).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            activation,
            activation_derivative,
            biases: DMatrix::from_row_slice(1, neurons, &biases),
            deltas: DMatrix::zeros(0, 0),
            errors: DMatrix::zeros(0, 0),
            last_activated_output: DMatrix::identity(1, 1),
            last_raw_output: DMatrix::identity(1, 1),
            optimizer_params: HashMap::new(),
            weights: DMatrix::from_row_slice(input_dim, neurons, &weights),
        }
    }

    pub fn from(
        activation: fn(f64) -> f64,
        activation_derivative: fn(f64) -> f64,
        biases: DMatrix<f64>,
        weights: DMatrix<f64>,
    ) -> Self {
        Self {
            activation,
            activation_derivative,
            biases,
            deltas: DMatrix::zeros(0, 0),
            errors: DMatrix::zeros(0, 0),
            last_activated_output: DMatrix::identity(1, 1),
            last_raw_output: DMatrix::identity(1, 1),
            optimizer_params: HashMap::new(),
            weights,
        }
    }

    pub fn forward(&mut self, data: &DMatrix<f64>) -> &DMatrix<f64> {
        self.last_raw_output = (data * &self.weights) + &self.biases;

        self.last_activated_output = self.last_raw_output.map(|x| (self.activation)(x));

        &self.last_activated_output
    }

    pub fn propagate_error(
        &mut self,
        last_layer: bool,
        next_layer_delta: &DMatrix<f64>,
        next_layer_weights: &DMatrix<f64>,
        previous_layer_output: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let activation_derivative = self
            .last_raw_output
            .map(|x| (self.activation_derivative)(x));

        let delta = if last_layer {
            activation_derivative
                .component_mul(&next_layer_delta)
                .transpose()
        } else {
            let next_layer_weighted_error = next_layer_weights * next_layer_delta;

            activation_derivative
                .transpose()
                .component_mul(&next_layer_weighted_error)
        };

        if self.errors.is_empty() {
            self.errors = (&delta * previous_layer_output).transpose();
        } else {
            self.errors += (&delta * previous_layer_output).transpose();
        }

        if self.deltas.is_empty() {
            self.deltas = delta.transpose();
        } else {
            self.deltas += delta.transpose();
        }

        delta
    }

    fn clear_error_and_delta(&mut self) {
        self.errors = DMatrix::zeros(0, 0);
        self.deltas = DMatrix::zeros(0, 0);
    }

    pub fn get_optimizer_params(&mut self) -> &mut HashMap<String, DMatrix<f64>> {
        &mut self.optimizer_params
    }

    pub fn get_last_output(&self) -> DMatrix<f64> {
        self.last_activated_output.clone()
    }

    pub fn get_biases(&self) -> DMatrix<f64> {
        self.biases.clone()
    }

    pub fn get_deltas_clone(&self) -> DMatrix<f64> {
        self.deltas.clone()
    }

    pub fn get_errors_clone(&self) -> DMatrix<f64> {
        self.errors.clone()
    }

    pub fn get_biases_reference(&mut self) -> &mut DMatrix<f64> {
        &mut self.biases
    }

    pub fn get_weights_reference(&mut self) -> &mut DMatrix<f64> {
        &mut self.weights
    }

    pub fn get_weights(&self) -> DMatrix<f64> {
        self.weights.clone()
    }

    pub fn get_input_dim(&self) -> usize {
        self.weights.shape().0
    }

    pub fn get_output_dim(&self) -> usize {
        self.weights.shape().1
    }
}
