use std::env;

use functions::{
    activations::{sigmoid, sigmoid_derivative},
    losses::{mse, mse_derivative},
};
use nalgebra::DMatrix;

use crate::core::{layer::Layer, model::Model};

mod core;
mod functions;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let hidden_layer = Layer::new(sigmoid, sigmoid_derivative, 2, 3);

    let output_layer = Layer::new(sigmoid, sigmoid_derivative, 3, 1);

    let mut model = Model::new(vec![hidden_layer, output_layer], mse, mse_derivative);

    model.fit(
        1,
        1000,
        0.01,
        vec![
            DMatrix::from_vec(1, 2, vec![0.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 1.0]),
        ],
        vec![
            DMatrix::from_vec(1, 1, vec![0.0]),
            DMatrix::from_vec(1, 1, vec![0.0]),
            DMatrix::from_vec(1, 1, vec![0.0]),
            DMatrix::from_vec(1, 1, vec![1.0]),
        ],
    )
}
