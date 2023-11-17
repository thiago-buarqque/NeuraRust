use nalgebra::DMatrix;

use super::optimizer::Optimizer;

pub struct RMSProp {
    decay_rate: f32,
}

impl Optimizer for RMSProp {
    fn initialize_layer_additional_params(&self, layer: &mut crate::core::layer::Layer) {
        let input_dim = layer.get_input_dim();
        let output_dim = layer.get_output_dim();

        let optimizer_params = layer.get_optimizer_params();

        optimizer_params.insert(
            "weights_moving_avg".to_string(),
            DMatrix::zeros(input_dim, output_dim),
        );
        optimizer_params.insert(
            "biases_moving_avg".to_string(),
            DMatrix::zeros(1, output_dim),
        );
    }

    fn update_params(
        &mut self,
        batch_size: usize,
        layer: &mut crate::core::layer::Layer,
        learning_rate: f32,
    ) {
        let (weights_moving_avg, biases_moving_avg) = self.calculate_moving_avg(layer);

        let mut errors = layer.get_errors_clone();

        let mut w_step_sizes = 
            weights_moving_avg.map(|x| learning_rate / (x + 1e-8).sqrt());

        w_step_sizes = w_step_sizes.component_mul(&errors);

        let weights_ref = layer.get_weights_reference();
        *weights_ref -= w_step_sizes.map(|x| x / batch_size as f32);

        let mut deltas = layer.get_deltas_clone();

        let mut b_step_sizes = 
        biases_moving_avg.map(|x| learning_rate / (x + 1e-8).sqrt());
        
        b_step_sizes = b_step_sizes.component_mul(&deltas);

        let biases_ref = layer.get_biases_reference();
        *biases_ref -= b_step_sizes.map(|x| x / batch_size as f32);
    }
}

impl RMSProp {
    pub fn new(decay_rate: f32) -> Self {
        Self { decay_rate }
    }
    fn calculate_moving_avg(
        &mut self,
        layer: &mut crate::core::layer::Layer,
    ) -> (DMatrix<f32>, DMatrix<f32>) {
        let mut errors = layer.get_errors_clone();
        let mut deltas = layer.get_deltas_clone();

        let optimizer_params = layer.get_optimizer_params();

        let mut weights_moving_avg = match optimizer_params.get("weights_moving_avg") {
            Some(data) => data.clone(),
            None => DMatrix::zeros(0, 0),
        };

        weights_moving_avg.scale_mut(self.decay_rate);

        errors = errors.map(|x| x * x);
        errors.scale_mut(1.0 - self.decay_rate);

        weights_moving_avg += errors;

        optimizer_params.insert("weights_moving_avg".to_string(), weights_moving_avg.clone());

        let mut biases_moving_avg = match optimizer_params.get("biases_moving_avg") {
            Some(data) => data.clone(),
            None => DMatrix::zeros(0, 0),
        };

        biases_moving_avg.scale_mut(self.decay_rate);

        deltas = deltas.map(|x| x * x);
        deltas.scale_mut(1.0 - self.decay_rate);

        biases_moving_avg += deltas;

        optimizer_params.insert("biases_moving_avg".to_string(), biases_moving_avg.clone());

        (weights_moving_avg, biases_moving_avg)
    }
}
