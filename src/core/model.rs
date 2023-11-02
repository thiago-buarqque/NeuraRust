use nalgebra::DMatrix;
use rand::{thread_rng, Rng};

use super::layer::Layer;

pub struct Model {
    layers: Vec<Layer>,
    loss: fn(&DMatrix<f64>, &DMatrix<f64>) -> f64,
    loss_derivative: fn(&DMatrix<f64>, &DMatrix<f64>) -> DMatrix<f64>,
}

impl Model {
    pub fn new(
        layers: Vec<Layer>,
        loss: fn(&DMatrix<f64>, &DMatrix<f64>) -> f64,
        loss_derivative: fn(&DMatrix<f64>, &DMatrix<f64>) -> DMatrix<f64>,
    ) -> Self {
        Self {
            layers,
            loss,
            loss_derivative,
        }
    }

    fn shuffle_dataset<T>(x: &mut [T], y: &mut [T]) {
        let mut rng = thread_rng();
        for i in (1..x.len()).rev() {
            let j = rng.gen_range(0..=i);
            x.swap(i, j);
            y.swap(i, j);
        }
    }

    pub fn fit(
        &mut self,
        batch_size: usize,
        epochs: usize,
        learning_rate: f64,
        x: Vec<DMatrix<f64>>,
        y: Vec<DMatrix<f64>>,
    ) {
        let mut x = x.clone();
        let mut y = y.clone();

        for epoch in 0..epochs {
            Self::shuffle_dataset(&mut x, &mut y);

            // Create batches
            let batches = x.chunks(batch_size).zip(y.chunks(batch_size));

            let mut epoch_loss = 0_f64;
            for (input_batch, target_batch) in batches {
                self.layers
                    .iter_mut()
                    .for_each(|layer| layer.clear_error_and_delta());

                let mut batch_loss = 0_f64;

                // Process each batch
                for (input_data, target_data) in input_batch.iter().zip(target_batch.iter()) {
                    let prediction = self.evaluate(input_data);

                    batch_loss += (self.loss)(target_data, &prediction);

                    self.backpropagation(target_data, input_data, &prediction);
                }

                // Update parameters based on batch
                self.layers
                    .iter_mut()
                    .for_each(|layer| layer.update_params(learning_rate, input_batch.len()));

                // Average loss for the batch
                epoch_loss += batch_loss / input_batch.len() as f64;
            }

            // Report epoch loss
            println!(
                "({}) Loss: {}",
                epoch,
                epoch_loss / (x.len() as f64 / batch_size as f64)
            );
        }
    }

    fn backpropagation(
        &mut self,
        expected: &DMatrix<f64>,
        network_input: &DMatrix<f64>,
        predicted: &DMatrix<f64>,
    ) {
        // TODO Initalize empty
        let mut next_layer_delta = (self.loss_derivative)(&expected, &predicted);

        for i in (0..self.layers.len()).rev() {
            let previous_layer_output = if i == 0 {
                network_input.clone()
            } else {
                self.layers[i - 1].get_last_output()
            };

            let last_layer = i == self.layers.len() - 1;

            let next_layer_weights = if last_layer {
                network_input.clone() // Could be empty, it won't be used anyway
            } else {
                self.layers[i + 1].get_weights()
            };

            next_layer_delta = self.layers[i].propagate_error(
                last_layer,
                &next_layer_delta,
                &next_layer_weights,
                &previous_layer_output,
            );
        }
    }

    pub fn evaluate(&mut self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let mut last_output = data.clone();

        self.layers
            .iter_mut()
            .for_each(|layer| last_output = layer.forward(&last_output));

        last_output
    }
}
