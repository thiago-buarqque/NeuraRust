use std::collections::HashMap;

use rand::{Rng, rngs::StdRng, SeedableRng};

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
        let mut r = rand::thread_rng();
        let mut r = StdRng::seed_from_u64(222);

        let weights: Vec<f32> = (0..input_dim * neurons)
            .map(|_| r.gen_range(-0.5..0.5))
            .collect();

        let biases: Vec<f32> = (0..neurons).map(|_| r.gen_range(-0.5..0.5)).collect();

        Self {
            activation,
            activation_derivative,
            biases: DMatrix::from_row_slice(neurons, 1, &biases),
            deltas: DMatrix::zeros(0, 0),
            errors: DMatrix::zeros(0, 0),
            last_activated_output: DMatrix::identity(1, 1),
            last_raw_output: DMatrix::identity(1, 1),
            optimizer_params: HashMap::new(),
            weights: DMatrix::from_row_slice(neurons, input_dim , &weights),
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
            deltas: DMatrix::zeros(0, 0),
            errors: DMatrix::zeros(0, 0),
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
        &mut self,
        last_layer: bool,
        next_layer_delta: &DMatrix<f32>,
        next_layer_weights: &DMatrix<f32>,
        previous_layer_output: &DMatrix<f32>,
    ) -> DMatrix<f32> {
        // https://sudeepraja.github.io/Neural/
        let activation_derivative = (self.activation_derivative)(&self.last_raw_output);

        let deltas = if last_layer {
            activation_derivative
                .component_mul(&next_layer_delta)
        } else {
            let next_layer_weighted_error = next_layer_weights.transpose() * next_layer_delta;

            next_layer_weighted_error
                .component_mul(&activation_derivative)
        };

        if self.errors.is_empty() {
            self.errors = (&deltas * previous_layer_output.transpose());
        } else {
            self.errors += (&deltas * previous_layer_output.transpose());
        }

        if self.deltas.is_empty() {
            self.deltas = deltas.clone();
        } else {
            self.deltas += deltas.clone();
        }

        deltas
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
        self.errors = DMatrix::zeros(0, 0);
        self.deltas = DMatrix::zeros(0, 0);
    }

    pub fn get_optimizer_params_mut_reference(&mut self) -> &mut HashMap<String, DMatrix<f32>> {
        &mut self.optimizer_params
    }

    pub fn get_optimizer_params_reference(&self) -> &HashMap<String, DMatrix<f32>> {
        &self.optimizer_params
    }

    pub fn get_last_output(&self) -> DMatrix<f32> {
        self.last_activated_output.clone()
    }

    pub fn get_biases(&self) -> DMatrix<f32> {
        self.biases.clone()
    }

    pub fn get_deltas_clone(&self) -> DMatrix<f32> {
        self.deltas.clone()
    }

    pub fn get_errors_clone(&self) -> DMatrix<f32> {
        self.errors.clone()
    }

    pub fn get_biases_reference(&mut self) -> &mut DMatrix<f32> {
        &mut self.biases
    }

    pub fn get_weights_reference(&mut self) -> &mut DMatrix<f32> {
        &mut self.weights
    }

    pub fn get_weights(&self) -> DMatrix<f32> {
        self.weights.clone()
    }

    pub fn get_input_dim(&self) -> usize {
        self.weights.shape().1
    }

    pub fn get_output_dim(&self) -> usize {
        self.weights.shape().0
    }
}
