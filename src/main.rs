#![allow(warnings)]
use std::time::Instant;
use std::{env, fs::File};

use csv::{self, Writer};
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
        losses::{
            binary_crossentropy, binary_crossentropy_derivative, categorical_crossentropy,
            categorical_crossentropy_derivative,
        },
    },
};

mod core;
mod functions;
mod optimizers;

fn read_csv(file_path: &str) -> Result<(Vec<DMatrix<f32>>, Vec<DMatrix<f32>>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(file_path)?;

    let mut x: Vec<DMatrix<f32>> = Vec::new();
    let mut y: Vec<DMatrix<f32>> = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let class = record[0].parse::<usize>().unwrap();

        let float_record: Vec<f32> = record
            .iter()
            .map(|x| x.parse::<f32>().unwrap() / 255.0)
            .collect();

        x.push(DMatrix::from_vec(
            float_record.len() - 1,
            1,
            float_record[1..].to_vec(),
        ));

        let mut label = DMatrix::zeros(1, 10);

        label[class] = 1.0;

        y.push(label.transpose());
    }

    Ok((x, y))
}

fn write_csv(path: &str, data: Vec<Vec<String>>) -> Result<(), Box<dyn Error>> {
    let file_path = path;
    let file = File::create(file_path)?;
    let mut wtr = Writer::from_writer(file);

    for row in data {
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let file_path = "./mnist/mnist_train.csv";

    let now = Instant::now();

    match read_csv(file_path) {
        Ok((x, y)) => {
            println!("Loaded data x = {} y = {})", x.len(), y.len());
            println!("y = {:?}", y[0]);

            let hidden_layer1 = Layer::new(relu, relu_derivative, x[0].len(), 512);

            let hidden_layer2 = Layer::new(relu, relu_derivative, 512, 512);

            // let hidden_layer3 = Layer::new(relu, relu_derivative, 1024, 512);

            // let hidden_layer4 = Layer::new(relu, relu_derivative, 512, 128);

            let output_layer = Layer::new(softmax, softmax_derivative, 512, y[0].len());

            let mut model = Model::new(
                vec![
                    hidden_layer1,
                    hidden_layer2,
                    // hidden_layer3,
                    // hidden_layer4,
                    output_layer,
                ],
                categorical_crossentropy,
                categorical_crossentropy_derivative,
            );

            let mut rmsprop = RMSProp::new(0.9);

            model.fit(
                128,
                10,
                0.001,
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

    // let hidden_layer = Layer::new(relu, relu_derivative, 2, 20);

    // let output_layer = Layer::new(sigmoid, sigmoid_derivative, 20, 1);

    // let mut model = Model::new(
    //     vec![hidden_layer, output_layer],
    //     binary_crossentropy,
    //     binary_crossentropy_derivative,
    // );

    // let mut rmsprop = RMSProp::new(0.9);

    // model.fit(
    //     1,
    //     1000,
    //     0.001,
    //     vec!["accuracy".to_string()],
    //     &mut rmsprop,
    //     vec![
    //         DMatrix::from_vec(2, 1, vec![0.0, 0.0]),
    //         DMatrix::from_vec(2, 1, vec![1.0, 0.0]),
    //         DMatrix::from_vec(2, 1, vec![0.0, 1.0]),
    //         DMatrix::from_vec(2, 1, vec![1.0, 1.0]),
    //     ],
    //     vec![
    //         DMatrix::from_vec(1, 1, vec![0.0]),
    //         DMatrix::from_vec(1, 1, vec![1.0]),
    //         DMatrix::from_vec(1, 1, vec![1.0]),
    //         DMatrix::from_vec(1, 1, vec![0.0]),
    //     ],
    // );

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}
