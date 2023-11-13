#![allow(warnings)]
use std::env;
use std::time::Instant;

use csv;
use rayon::prelude::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use std::error::Error;
use std::io;

use functions::{
    activations::{sigmoid, sigmoid_derivative},
    losses::{mse, mse_derivative},
};
use nalgebra::DMatrix;
use optimizers::rmsprop::RMSProp;

use crate::{
    core::{layer::Layer, model::Model},
    functions::{
        activations::{relu, relu_derivative},
        losses::{categorical_crossentropy, categorical_crossentropy_derivative},
    },
};

mod core;
mod functions;
mod optimizers;

fn read_csv(file_path: &str) -> Result<(Vec<DMatrix<f64>>, Vec<DMatrix<f64>>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(file_path)?;

    let mut x: Vec<DMatrix<f64>> = Vec::new();
    let mut y: Vec<DMatrix<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let float_record: Vec<f64> = record
            .iter()
            .map(|x| x.parse::<f64>().unwrap() / 255.0)
            .collect();

        x.push(DMatrix::from_vec(
            1,
            float_record.len() - 1,
            float_record[1..].to_vec(),
        ));

        let mut label = DMatrix::zeros(1, 10);

        label[float_record[0] as usize] = 1.0;

        y.push(label);
    }

    Ok((x, y))
}
fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let file_path = "/home/evry/Desktop/repositories/neura_rust/mnist/mnist_test.csv"; // Replace with your file path

    match read_csv(file_path) {
        Ok((x, y)) => {
            println!("Loaded data x = {} y = {})", x.len(), y.len());

            let hidden_layer1 = Layer::new(relu, relu_derivative, x[0].len(), 128);

            let hidden_layer2 = Layer::new(relu, relu_derivative, 128, 64);

            let hidden_layer3 = Layer::new(relu, relu_derivative, 64, 64);

            let output_layer = Layer::new(sigmoid, sigmoid_derivative, 64, y[0].len());

            let mut model = Model::new(
                vec![hidden_layer1, hidden_layer2, hidden_layer3, output_layer],
                categorical_crossentropy,
                categorical_crossentropy_derivative,
            );

            let mut rmsprop = RMSProp::new(0.9);

            model.fit(
                128,
                50,
                0.05,
                vec![
                    "accuracy".to_string(),
                    "recall".to_string(),
                    "f1-score".to_string(),
                    "precision".to_string(),
                ],
                &mut rmsprop,
                x,
                y,
            );
        }
        Err(_) => println!("Error reading CSV file:"),
    }

    // let hidden_layer = Layer::new(sigmoid, sigmoid_derivative, 2, 32);

    // let output_layer = Layer::new(sigmoid, sigmoid_derivative, 32, 1);

    // let mut model = Model::new(vec![hidden_layer, output_layer], mse, mse_derivative);

    // let mut rmsprop = RMSProp::new(0.9);

    // let now = Instant::now();

    // model.fit(
    //     1,
    //     100,
    //     0.01,
    //     vec!["accuracy".to_string(), "recall".to_string()],
    //     &mut rmsprop,
    //     vec![
    //         DMatrix::from_vec(1, 2, vec![0.0, 0.0]),
    //         DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
    //         DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
    //         DMatrix::from_vec(1, 2, vec![1.0, 1.0]),
    //     ],
    //     vec![
    //         DMatrix::from_vec(1, 1, vec![0.0]),
    //         DMatrix::from_vec(1, 1, vec![1.0]),
    //         DMatrix::from_vec(1, 1, vec![1.0]),
    //         DMatrix::from_vec(1, 1, vec![0.0]),
    //     ],
    // );

    // let elapsed = now.elapsed();
    // println!("Elapsed: {:.2?}", elapsed);
}
