use std::fmt::Write;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nalgebra::DMatrix;
use rand::{thread_rng, Rng};

use crate::{functions::metrics::print_metrics, optimizers::optimizer::Optimizer};

use super::layer::Layer;

pub struct Model {
    layers: Vec<Layer>,
    loss: fn(&DMatrix<f32>, &DMatrix<f32>) -> f32,
    loss_derivative: fn(&DMatrix<f32>, &DMatrix<f32>) -> DMatrix<f32>,
}

impl Model {
    pub fn new(
        layers: Vec<Layer>,
        loss: fn(&DMatrix<f32>, &DMatrix<f32>) -> f32,
        loss_derivative: fn(&DMatrix<f32>, &DMatrix<f32>) -> DMatrix<f32>,
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
        learning_rate: f32,
        metrics: Vec<String>,
        optimizer: &mut dyn Optimizer,
        x: Vec<DMatrix<f32>>,
        y: Vec<DMatrix<f32>>,
    ) {
        let mut x = x.clone();
        let mut y = y.clone();

        self.layers
            .iter_mut()
            .for_each(|layer| optimizer.initialize_layer_additional_params(layer));

        for epoch in 0..epochs {
            Self::shuffle_dataset(&mut x, &mut y);

            let batches = x.chunks(batch_size).zip(y.chunks(batch_size));

            let mut epoch_predictions = Vec::with_capacity(x.len());
            let mut epoch_loss = 0_f32;

            // let progress_bar = ProgressBar::new(batches.len() as u64);

            // progress_bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            //     .unwrap()
            //     .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap())
            //     .progress_chars("#>-"));

            let mut first = true;
            for (input_batch, target_batch) in batches {
                // progress_bar.inc(1);

                let mut batch_loss = 0_f32;

                self.layers
                    .iter_mut()
                    .for_each(|layer| layer.clear_error_and_delta());

                for (input_data, target_data) in input_batch.iter().zip(target_batch.iter()) {
                    let mut prediction = self.evaluate(input_data);

                    // if prediction.len() == 1 {
                    //     // Binary classification
                    //     prediction = prediction.map(|x| if x > 0.5 { 1.0 } else { 0.0 })
                    // }

                    batch_loss += (self.loss)(target_data, &prediction);

                    self.backpropagation(target_data, input_data, &prediction);

                    epoch_predictions.push(prediction);
                }

                self.layers.iter_mut().for_each(|layer| {
                    optimizer.update_params(input_batch.len(), layer, learning_rate)
                });

                epoch_loss += batch_loss as f32;
                // progress_bar.set_message(format!("Batch loss: {}", batch_loss));
            }

            // progress_bar.finish();

            if epoch % 5 == 0 {
                print!("({}) Loss: {} ", epoch, epoch_loss / x.len() as f32);

                print_metrics(epoch_predictions, &metrics, &y);
                println!()
            }
        }
    }

    fn backpropagation(
        &mut self,
        expected: &DMatrix<f32>,
        network_input: &DMatrix<f32>,
        predicted: &DMatrix<f32>,
    ) {
        let mut next_layer_delta = (self.loss_derivative)(&expected, &predicted);

        for i in (0..self.layers.len()).rev() {
            let previous_layer_output = if i == 0 {
                network_input.clone()
            } else {
                self.layers[i - 1].get_last_output()
            };

            let last_layer = i == self.layers.len() - 1;

            let next_layer_weights = if last_layer {
                DMatrix::zeros(0, 0)
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

    pub fn evaluate(&mut self, data: &DMatrix<f32>) -> DMatrix<f32> {
        let mut last_output = data;

        self.layers
            .iter_mut()
            .for_each(|layer| last_output = layer.forward(&last_output));

        last_output.clone()
    }
}
