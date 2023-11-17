use nalgebra::DMatrix;

use crate::core::layer::Layer;

pub trait Optimizer {
    fn initialize_layer_additional_params(&self, layer: &mut Layer);
    fn update_params(&mut self, batch_size: usize, layer: &mut Layer, learning_rate: f32);
}
