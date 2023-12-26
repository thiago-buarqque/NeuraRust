use std::collections::HashMap;

use rand_distr::{Distribution, Normal};

use nalgebra::DMatrix;

pub struct Layer {
    activation: fn(&DMatrix<f32>) -> DMatrix<f32>,
    activation_derivative: fn(&DMatrix<f32>) -> DMatrix<f32>,
    biases: DMatrix<f32>,
    deltas: DMatrix<f32>,
    errors: DMatrix<f32>,
    last_activated_output: DMatrix<f32>,
    last_raw_output: DMatrix<f32>,
    optimizer_params: HashMap<String, DMatrix<f32>>,
    weights: DMatrix<f32>,
}

impl Layer {
    pub fn new(
        activation: fn(&DMatrix<f32>) -> DMatrix<f32>,
        activation_derivative: fn(&DMatrix<f32>) -> DMatrix<f32>,
        input_dim: usize,
        neurons: usize,
    ) -> Self {
        // let mut r = StdRng::seed_from_u64(222);

        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();

        let weights_sample: Vec<f32> = normal
            .sample_iter(&mut rand::thread_rng())
            .take(input_dim * neurons)
            .collect();

        let std = (2.0_f32 / input_dim as f32).sqrt();

        let weights: Vec<f32> = weights_sample.iter().map(|x| x * std).collect();

        let biases_sample: Vec<f32> = normal
            .sample_iter(&mut rand::thread_rng())
            .take(neurons)
            .collect();

        let biases: Vec<f32> = biases_sample.iter().map(|x| x * std).collect();

        Self {
            activation,
            activation_derivative,
            biases: DMatrix::from_row_slice(neurons, 1, &biases),
            deltas: DMatrix::zeros(neurons, 1),
            errors: DMatrix::zeros(neurons, input_dim),
            last_activated_output: DMatrix::identity(1, 1),
            last_raw_output: DMatrix::identity(1, 1),
            optimizer_params: HashMap::new(),
            weights: DMatrix::from_row_slice(neurons, input_dim, &weights),
        }
    }

    pub fn from(
        activation: fn(&DMatrix<f32>) -> DMatrix<f32>,
        activation_derivative: fn(&DMatrix<f32>) -> DMatrix<f32>,
        biases: DMatrix<f32>,
        weights: DMatrix<f32>,
    ) -> Self {
        Self {
            activation,
            activation_derivative,
            biases,
            deltas: DMatrix::zeros(weights.nrows(), 1),
            errors: DMatrix::zeros(weights.nrows(), weights.ncols()),
            last_activated_output: DMatrix::identity(1, 1),
            last_raw_output: DMatrix::identity(1, 1),
            optimizer_params: HashMap::new(),
            weights,
        }
    }

    pub fn forward(&mut self, data: &DMatrix<f32>) -> &DMatrix<f32> {
        self.last_raw_output = (&self.weights * data) + &self.biases;

        self.last_activated_output = (self.activation)(&self.last_raw_output);

        &self.last_activated_output
    }

    pub fn propagate_error(
        &self,
        last_layer: bool,
        next_layer_delta: &DMatrix<f32>,
        next_layer_weights: &DMatrix<f32>,
        previous_layer_output: &DMatrix<f32>,
    ) -> (DMatrix<f32>, DMatrix<f32>) {
        let activation_derivative = (self.activation_derivative)(&self.last_raw_output);

        let deltas = if last_layer {
            activation_derivative.component_mul(&next_layer_delta)
        } else {
            let next_layer_weighted_error = next_layer_weights.transpose() * next_layer_delta;

            next_layer_weighted_error.component_mul(&activation_derivative)
        };

        (&deltas * previous_layer_output.transpose(), deltas)
    }

    pub fn sum_errors_and_deltas(&mut self, deltas: &DMatrix<f32>, errors: &DMatrix<f32>) {
        self.errors += errors;
        self.deltas += deltas;
    }

    pub fn update_params(&mut self, learning_rate: f32, batch_size: usize) {
        let mut error = &self.errors / batch_size as f32;

        error.scale_mut(learning_rate);

        self.weights -= error;

        let mut deltas = &self.deltas / batch_size as f32;

        deltas.scale_mut(learning_rate);

        self.biases -= deltas;

        self.clear_error_and_delta()
    }

    pub fn clear_error_and_delta(&mut self) {
        self.deltas = DMatrix::zeros(self.weights.nrows(), 1);
        self.errors = DMatrix::zeros(self.weights.nrows(), self.weights.ncols());
    }

    pub fn get_optimizer_params_mut_reference(&mut self) -> &mut HashMap<String, DMatrix<f32>> {
        &mut self.optimizer_params
    }

    pub fn get_optimizer_params_reference(&self) -> &HashMap<String, DMatrix<f32>> {
        &self.optimizer_params
    }

    pub fn get_last_output(&self) -> &DMatrix<f32> {
        &self.last_activated_output
    }

    pub fn get_deltas_clone(&self) -> DMatrix<f32> {
        self.deltas.clone()
    }

    pub fn get_errors_clone(&self) -> DMatrix<f32> {
        self.errors.clone()
    }

    pub fn get_biases_mut_reference(&mut self) -> &mut DMatrix<f32> {
        &mut self.biases
    }

    pub fn get_weights_mut_reference(&mut self) -> &mut DMatrix<f32> {
        &mut self.weights
    }

    pub fn get_weights_reference(&self) -> &DMatrix<f32> {
        &self.weights
    }

    pub fn get_input_dim(&self) -> usize {
        self.weights.shape().1
    }

    pub fn get_output_dim(&self) -> usize {
        self.weights.shape().0
    }
}
