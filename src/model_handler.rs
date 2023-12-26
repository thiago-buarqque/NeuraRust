use csv::{self};
use std::error::Error;
use std::str::FromStr;

use nalgebra::DMatrix;

use crate::{
    core::{layer::Layer, model::Model},
    functions::{
        activations::{relu, relu_derivative, softmax, softmax_derivative},
        losses::{categorical_crossentropy, categorical_crossentropy_derivative},
    }, optimizers::rmsprop::RMSProp,
};


// Read MNIST dataset
// fn read_mnist(file_path: &str) -> Result<(Vec<DMatrix<f32>>, Vec<DMatrix<f32>>), Box<dyn Error>> {
//     let mut rdr = csv::Reader::from_path(file_path)?;

//     let mut x: Vec<DMatrix<f32>> = Vec::new();
//     let mut y: Vec<DMatrix<f32>> = Vec::new();

//     for result in rdr.records() {
//         let record = result?;

//         let class = record[0].parse::<usize>().unwrap();

//         let float_record: Vec<f32> = record
//             .iter()
//             .map(|x| x.parse::<f32>().unwrap() / 255.0)
//             .collect();

//         x.push(DMatrix::from_vec(
//             float_record.len() - 1,
//             1,
//             float_record[1..].to_vec(),
//         ));

//         let mut label = DMatrix::zeros(1, 10);

//         label[class] = 1.0;

//         y.push(label.transpose());
//     }

//     Ok((x, y))
// }

// Read Google's Quick, Draw Doodles dataset (personal file)
fn read_doodles(file_path: &str) -> Result<(Vec<DMatrix<f32>>, Vec<DMatrix<f32>>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(file_path)?;

    let mut x: Vec<DMatrix<f32>> = Vec::new();
    let mut y: Vec<DMatrix<f32>> = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let class = record[1].parse::<usize>().unwrap();

        let numbers: Vec<f32> = record[2]
            .trim_matches(|p| p == '[' || p == ']')
            .split(',')
            .map(|s| f32::from_str(s.trim()).unwrap() / 255.0)
            .collect();

        x.push(DMatrix::from_vec(numbers.len(), 1, numbers));

        let mut label = DMatrix::zeros(1, 10);

        label[class] = 1.0;

        y.push(label.transpose());
    }

    Ok((x, y))
}

pub fn get_trained_model() -> Option<Model>{
    match (
        read_doodles("./doodles/train-quick-draw.csv"),
        read_doodles("./doodles/test-quick-draw.csv"),
    ) {
        (Ok((x_train, y_train)), Ok((x_test, y_test))) => {
            println!(
                "Loaded data x_train = {} y_train = {} x_test = {} y_test = {})",
                x_train.len(),
                y_train.len(),
                x_test.len(),
                y_test.len()
            );
            // println!("x[0] = {}", x_train[0]);

            let hidden_layer1 = Layer::new(relu, relu_derivative, x_train[0].len(), 1024);

            let hidden_layer2 = Layer::new(relu, relu_derivative, 1024, 512);

            let output_layer = Layer::new(softmax, softmax_derivative, 512, y_train[0].len());

            let mut model = Model::new(
                vec![hidden_layer1, hidden_layer2, output_layer],
                categorical_crossentropy,
                categorical_crossentropy_derivative,
            );

            let mut rmsprop = RMSProp::new(0.9);

            let metrics = vec![
                "accuracy".to_string(),
                "recall".to_string(),
                "f1-score".to_string(),
                "precision".to_string(),
            ];

            model.fit(
                64,
                10,
                0.001,
                metrics.clone(),
                &mut rmsprop,
                x_train,
                y_train,
            );

            println!("\nTesting the network:\n");

            model.test(metrics, &x_test, &y_test);

            Some(model)
        }
        _ => {
            println!("Error reading CSV file:");
            None
        }
    }
}