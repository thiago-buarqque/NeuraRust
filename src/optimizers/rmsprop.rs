use std::collections::HashMap;

use nalgebra::DMatrix;

use super::optimizer::Optimizer;

pub struct RMSProp {
    decay_rate: f32,
}

impl Optimizer for RMSProp {
    fn initialize_layer_additional_params(&self, layer: &mut crate::core::layer::Layer) {
        let input_dim = layer.get_input_dim();
        let output_dim = layer.get_output_dim();

        let optimizer_params = layer.get_optimizer_params_mut_reference();

        optimizer_params.insert(
            "weights_moving_avg".to_string(),
            DMatrix::zeros(output_dim, input_dim),
        );
        optimizer_params.insert(
            "biases_moving_avg".to_string(),
            DMatrix::zeros(output_dim, 1),
        );

        optimizer_params.insert(
            "weights_momentum".to_string(),
            DMatrix::zeros(output_dim, input_dim),
        );
        optimizer_params.insert(
            "biases_momentum".to_string(),
            DMatrix::zeros(output_dim, 1),
        );
    }

    fn update_params(
        &mut self,
        batch_size: usize,
        layer: &mut crate::core::layer::Layer,
        learning_rate: f32,
    ) {
        self.calculate_moving_avg(layer);

        let mut errors = layer.get_errors_clone();
        let mut deltas = layer.get_deltas_clone();

        let (weights_moving_avg, biases_moving_avg, weights_momentum) = {
            let optimizer_params = layer.get_optimizer_params_reference();
            (
                optimizer_params.get("weights_moving_avg").unwrap(),
                optimizer_params.get("biases_moving_avg").unwrap(),
                optimizer_params.get("weights_momentum").unwrap(),
            )
        };

        let mut w_step_sizes = weights_moving_avg.map(|x| learning_rate / (x + 1e-8).sqrt());
        let mut b_step_sizes = biases_moving_avg.map(|x| learning_rate / (x + 1e-8).sqrt());

        // w_step_sizes = w_step_sizes.component_mul(&weights_momentum);
        w_step_sizes = w_step_sizes.component_mul(&errors);

        let weights_ref = layer.get_weights_mut_reference();
        *weights_ref -= w_step_sizes.map(|x| x / batch_size as f32);

        // b_step_sizes = b_step_sizes.component_mul(&layer.get_optimizer_params_reference().get("biases_momentum").unwrap());
        b_step_sizes = b_step_sizes.component_mul(&deltas);

        let biases_ref = layer.get_biases_mut_reference();
        *biases_ref -= b_step_sizes.map(|x| x / batch_size as f32);
    }
}

impl RMSProp {
    pub fn new(decay_rate: f32) -> Self {
        Self { decay_rate }
    }
    fn calculate_moving_avg(&mut self, layer: &mut crate::core::layer::Layer) {
        let mut errors = layer.get_errors_clone();
        let mut deltas = layer.get_deltas_clone();

        let mut optimizer_params = layer.get_optimizer_params_mut_reference();

        self.update_moving_avg(&mut optimizer_params, "weights_moving_avg", &errors);

        self.update_moving_avg(&mut optimizer_params, "biases_moving_avg", &deltas);

        self.update_momentum(&mut optimizer_params, "weights_momentum", errors);

        self.update_momentum(&mut optimizer_params, "biases_momentum", deltas);
    }

    fn update_moving_avg(
        &self,
        optimizer_params: &mut HashMap<String, DMatrix<f32>>,
        key: &str,
        gradients: &DMatrix<f32>,
    ) {
        if let Some(mut moving_avg) = optimizer_params.remove(key) {
            moving_avg.scale_mut(self.decay_rate);

            let mut squared_gradients = gradients.map(|x| x.powi(2));
            squared_gradients.scale_mut(1.0 - self.decay_rate);

            moving_avg += squared_gradients;

            optimizer_params.insert(key.to_string(), moving_avg);
        }
    }

    fn update_momentum(
        &self,
        optimizer_params: &mut HashMap<String, DMatrix<f32>>,
        key: &str,
        mut gradients: DMatrix<f32>,
    ) {
        if let Some(mut momentum) = optimizer_params.remove(key) {
            momentum.scale_mut(self.decay_rate);

            gradients.scale_mut(1.0 - self.decay_rate);

            momentum += gradients;

            optimizer_params.insert(key.to_string(), momentum);
        }
    }
}
