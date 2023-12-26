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

            let batches_ammount = batches.len();
            let progress_bar = ProgressBar::new(batches_ammount as u64);

            progress_bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap())
                .progress_chars("#>-"));

            for (i, (input_batch, target_batch)) in batches.enumerate() {
                progress_bar.inc(1);

                let mut batch_loss = 0_f32;

                for (input_data, target_data) in input_batch.iter().zip(target_batch.iter()) {
                    let prediction = self.evaluate(input_data);

                    batch_loss += (self.loss)(target_data, &prediction);

                    self.backpropagation(target_data, input_data, &prediction);

                    epoch_predictions.push(prediction);
                }

                self.layers.iter_mut().for_each(|layer| {
                    optimizer.update_params(input_batch.len(), layer, learning_rate);
                    layer.clear_error_and_delta()
                });

                epoch_loss += batch_loss / input_batch.len() as f32;
                progress_bar.set_message(format!("Loss: {:.4}", epoch_loss / (i + 1) as f32));
            }

            progress_bar.finish();

            print!("({}) Loss: {:.4} ", epoch, epoch_loss / batches_ammount as f32);

            print_metrics(epoch_predictions, &metrics, &y);
            println!()
        }
    }

    fn backpropagation(
        &mut self,
        expected: &DMatrix<f32>,
        network_input: &DMatrix<f32>,
        predicted: &DMatrix<f32>,
    ) {
        let mut next_layer_delta = (self.loss_derivative)(&expected, &predicted);

        let zeros = DMatrix::zeros(0, 0);

        let mut next_layer_errors;
        for i in (0..self.layers.len()).rev() {
            let previous_layer_output = if i == 0 {
                network_input
            } else {
                self.layers[i - 1].get_last_output()
            };

            let last_layer = i == self.layers.len() - 1;

            let next_layer_weights = if last_layer {
                &zeros
            } else {
                self.layers[i + 1].get_weights_reference()
            };

            (next_layer_errors, next_layer_delta) = self.layers[i].propagate_error(
                last_layer,
                &next_layer_delta,
                &next_layer_weights,
                &previous_layer_output,
            );

            self.layers[i].sum_errors_and_deltas(&next_layer_delta, &next_layer_errors)
        }
    }

    pub fn evaluate(&mut self, data: &DMatrix<f32>) -> DMatrix<f32> {
        let mut last_output = data;

        self.layers
            .iter_mut()
            .for_each(|layer| last_output = layer.forward(&last_output));

        last_output.clone()
    }

    pub fn test(&mut self, metrics: Vec<String>, x: &Vec<DMatrix<f32>>, y: &Vec<DMatrix<f32>>) {
        let mut loss = 0_f32;
        let mut predictions = Vec::with_capacity(x.len());

        for (_x, _y) in x.iter().zip(y.iter()) {
            let prediction = self.evaluate(_x);

            loss += (self.loss)(_y, &prediction);

            predictions.push(prediction);
        }

        print!("Loss: {:.4} ", loss / x.len() as f32);

        print_metrics(predictions, &metrics, &y);
        println!()
    }
}
