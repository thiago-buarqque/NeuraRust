#![allow(warnings)]
use std::sync::{Mutex, Arc};
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

fn read_mnist(file_path: &str) -> Result<(Vec<DMatrix<f32>>, Vec<DMatrix<f32>>), Box<dyn Error>> {
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

// fn write_csv(path: &str, data: Vec<Vec<String>>) -> Result<(), Box<dyn Error>> {
//     let file_path = path;
//     let file = File::create(file_path)?;
//     let mut wtr = Writer::from_writer(file);

//     for row in data {
//         wtr.write_record(&row)?;
//     }

//     wtr.flush()?;
//     Ok(())
// }

// fn main() {
//     env::set_var("RUST_BACKTRACE", "1");

//     let file_path = "./mnist/mnist_train.csv";

//     let now = Instant::now();

//     match read_csv(file_path) {
//         Ok((x, y)) => {
//             println!("Loaded data x = {} y = {})", x.len(), y.len());
//             println!("y = {:?}", y[0]);

//             let hidden_layer1 = Layer::new(relu, relu_derivative, x[0].len(), 512);

//             let hidden_layer2 = Layer::new(relu, relu_derivative, 512, 512);

//             // let hidden_layer3 = Layer::new(relu, relu_derivative, 1024, 512);

//             // let hidden_layer4 = Layer::new(relu, relu_derivative, 512, 128);

//             let output_layer = Layer::new(softmax, softmax_derivative, 512, y[0].len());

//             let mut model = Model::new(
//                 vec![
//                     hidden_layer1,
//                     hidden_layer2,
//                     // hidden_layer3,
//                     // hidden_layer4,
//                     output_layer,
//                 ],
//                 categorical_crossentropy,
//                 categorical_crossentropy_derivative,
//             );

//             let mut rmsprop = RMSProp::new(0.9);

//             model.fit(
//                 128,
//                 10,
//                 0.001,
//                 vec![
//                     "accuracy".to_string(),
//                     "recall".to_string(),
//                     "f1-score".to_string(),
//                     "precision".to_string(),
//                 ],
//                 &mut rmsprop,
//                 x,
//                 y,
//             );
//         }
//         Err(_) => println!("Error reading CSV file:"),
//     }

//     // let hidden_layer = Layer::new(relu, relu_derivative, 2, 20);

//     // let output_layer = Layer::new(sigmoid, sigmoid_derivative, 20, 1);

//     // let mut model = Model::new(
//     //     vec![hidden_layer, output_layer],
//     //     binary_crossentropy,
//     //     binary_crossentropy_derivative,
//     // );

//     // let mut rmsprop = RMSProp::new(0.9);

//     // model.fit(
//     //     1,
//     //     1000,
//     //     0.001,
//     //     vec!["accuracy".to_string()],
//     //     &mut rmsprop,
//     //     vec![
//     //         DMatrix::from_vec(2, 1, vec![0.0, 0.0]),
//     //         DMatrix::from_vec(2, 1, vec![1.0, 0.0]),
//     //         DMatrix::from_vec(2, 1, vec![0.0, 1.0]),
//     //         DMatrix::from_vec(2, 1, vec![1.0, 1.0]),
//     //     ],
//     //     vec![
//     //         DMatrix::from_vec(1, 1, vec![0.0]),
//     //         DMatrix::from_vec(1, 1, vec![1.0]),
//     //         DMatrix::from_vec(1, 1, vec![1.0]),
//     //         DMatrix::from_vec(1, 1, vec![0.0]),
//     //     ],
//     // );

//     let elapsed = now.elapsed();
//     println!("Elapsed: {:.2?}", elapsed);
// }

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct MatrixData {
    matrix: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct ApiResponse {
    message: String,
}

async fn upload_matrix(data: web::Json<MatrixData>, shared: web::Data<Arc<Mutex<Model>>>) -> impl Responder {
    let rows = data.matrix.len();
    let cols = if rows > 0 { data.matrix[0].len() } else { 0 };
    let mut matrix_elements = Vec::with_capacity(rows * cols);

    for row in &data.matrix {
        for &val in row {
            matrix_elements.push(val as f32);
        }
    }

    let dmatrix = DMatrix::from_row_slice(matrix_elements.len(), 1, &matrix_elements);
    println!("Matrix data: {:?}", data.matrix.to_vec());
    println!("Constructed matrix: {}", dmatrix);
    // Use `dmatrix` as needed
    let mut locked_model = shared.lock().unwrap();

    let prediction = locked_model.evaluate(&dmatrix);

    drop(locked_model);
    
    println!("Model predicted: {}", prediction);
    let prediction_vec = prediction.data.as_vec();
    let mut index_of_max: Option<usize> = prediction_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index);

    if index_of_max.is_none() {
        index_of_max = Some(99);
    }

    HttpResponse::Ok().json(ApiResponse {
        message: format!("Network predicted: {}", index_of_max.unwrap()),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env::set_var("RUST_BACKTRACE", "1");

    let file_path = "./mnist/mnist_train.csv";

    let now = Instant::now();

    let model = match (read_mnist(file_path), read_mnist("./mnist/mnist_test.csv")) {
        (Ok((x_train, y_train)), Ok(((x_test, y_test)))) => {
            println!("Loaded data x_train = {} y_train = {} x_test = {} y_test = {})", x_train.len(), y_train.len(), x_test.len(), y_test.len());
            println!("x[0] = {}", x_train[0]);

            let hidden_layer1 = Layer::new(relu, relu_derivative, x_train[0].len(), 512);

            let hidden_layer2 = Layer::new(relu, relu_derivative, 512, 512);

            // let hidden_layer3 = Layer::new(relu, relu_derivative, 1024, 64);

            // let hidden_layer4 = Layer::new(relu, relu_derivative, 64, 128);

            let output_layer = Layer::new(softmax, softmax_derivative, 512, y_train[0].len());

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

            let metrics = vec![
                    "accuracy".to_string(),
                    "recall".to_string(),
                    "f1-score".to_string(),
                    "precision".to_string(),
                ];
            
            model.fit(
                128,
                15,
                0.001,
                metrics.clone(),
                &mut rmsprop,
                x_train,
                y_train,
            );

            println!("\nTesting results:\n");

            model.test(metrics, &x_test, &y_test);

            Some(model)
        }
        _ => {
            println!("Error reading CSV file:");
            None
        },
    };

    let shared_data = Arc::new(Mutex::new(model.unwrap()));

    println!("Started server");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(web::Data::new(shared_data.clone()))
            .route("/upload", web::post().to(upload_matrix))
    })
    .bind("127.0.0.1:8000")?
    .run()
    .await
}
