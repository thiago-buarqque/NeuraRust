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
            weights,
        }
    }

    pub fn forward(&mut self, data: &DMatrix<f64>) -> DMatrix<f64> {
        // Data.column size must be equal to self.weights.lines size

        let mut output = (data * &self.weights) + &self.biases;

        self.last_raw_output = output.clone();

        output = output.map(|x| (self.activation)(x));

        self.last_activated_output = output.clone();

        output
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
            self.errors = &delta * previous_layer_output;
        } else {
            self.errors += &delta * previous_layer_output;
        }

        if self.deltas.is_empty() {
            self.deltas = delta.clone();
        } else {
            self.deltas += delta.clone();
        }

        delta
    }

    fn clear_error_and_delta(&mut self) {
        self.errors = DMatrix::zeros(0, 0);
        self.deltas = DMatrix::zeros(0, 0);
    }

    pub fn update_params(&mut self, learning_rate: f64, batch_size: usize) {
        let mut transposed_error = self.errors.transpose();

        transposed_error /= batch_size as f64;
        transposed_error.scale_mut(learning_rate);

        self.weights -= transposed_error;

        let mut transposed_delta = self.deltas.transpose();

        transposed_delta /= batch_size as f64;
        transposed_delta.scale_mut(learning_rate);

        self.biases -= transposed_delta;

        self.clear_error_and_delta()
    }

    pub fn get_last_output(&self) -> DMatrix<f64> {
        self.last_activated_output.clone()
    }

    pub fn get_weights(&self) -> DMatrix<f64> {
        self.weights.clone()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.weights.shape()
    }
}
