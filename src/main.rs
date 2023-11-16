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
        activations::{relu, relu_derivative, softmax, softmax_derivative},
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

        let class = record[0].parse::<usize>().unwrap();

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

        label[class] = 1.0;

        y.push(label);
    }

    Ok((x, y))
}
fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    // let file_path = "./mnist_test.csv";

    // match read_csv(file_path) {
    //     Ok((x, y)) => {
    //         println!("Loaded data x = {} y = {})", x.len(), y.len());

    //         let hidden_layer1 = Layer::new(relu, relu_derivative, x[0].len(), 256);

    //         let hidden_layer2 = Layer::new(relu, relu_derivative, 256, 128);

    //         let hidden_layer3 = Layer::new(relu, relu_derivative, 128, 128);

    //         let hidden_layer4 = Layer::new(sigmoid, sigmoid_derivative, 128, 128);

    //         let output_layer = Layer::new(softmax, softmax_derivative, 128, y[0].len());

    //         let mut model = Model::new(
    //             vec![hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4, output_layer],
    //             categorical_crossentropy,
    //             categorical_crossentropy_derivative,
    //         );

    //         let mut rmsprop = RMSProp::new(0.9);

    //         model.fit(
    //             256,
    //             100,
    //             0.05,
    //             vec![
    //                 "accuracy".to_string(),
    //                 "recall".to_string(),
    //                 "f1-score".to_string(),
    //                 "precision".to_string(),
    //             ],
    //             &mut rmsprop,
    //             x,
    //             y,
    //         );
    //     }
    //     Err(_) => println!("Error reading CSV file:"),
    // }

    let hidden_layer = Layer::new(relu, relu_derivative, 2, 256);

    let output_layer = Layer::new(softmax, softmax_derivative, 256, 2);

    let mut model = Model::new(vec![hidden_layer, output_layer], categorical_crossentropy, categorical_crossentropy_derivative);

    let mut rmsprop = RMSProp::new(0.9);

    let now = Instant::now();

    model.fit(
        1,
        50,
        0.01,
        vec!["accuracy".to_string()],
        &mut rmsprop,
        vec![
            DMatrix::from_vec(1, 2, vec![0.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 1.0]),
        ],
        vec![
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
        ],
    );

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}
