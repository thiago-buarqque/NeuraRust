#![allow(warnings)]
use std::env;
use std::sync::{Arc, Mutex};

use csv::{self};
use std::error::Error;
use std::str::FromStr;

use nalgebra::DMatrix;
use optimizers::rmsprop::RMSProp;

use crate::{
    core::{layer::Layer, model::Model},
    functions::{
        activations::{relu, relu_derivative, softmax, softmax_derivative},
        losses::{categorical_crossentropy, categorical_crossentropy_derivative},
    },
};

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

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

#[derive(Deserialize)]
struct MatrixData {
    matrix: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct ApiResponse {
    prediction: f32,
    accuracy: f32,
}

async fn upload_matrix(
    data: web::Json<MatrixData>,
    shared: web::Data<Arc<Mutex<Model>>>,
) -> impl Responder {
    let rows = data.matrix.len();
    let cols = if rows > 0 { data.matrix[0].len() } else { 0 };
    let mut matrix_elements = Vec::with_capacity(rows * cols);

    for row in &data.matrix {
        for &val in row {
            matrix_elements.push(val as f32);
        }
    }

    let dmatrix = DMatrix::from_row_slice(matrix_elements.len(), 1, &matrix_elements);
    // Use `dmatrix` as needed
    let mut locked_model = shared.lock().unwrap();

    let prediction = locked_model.evaluate(&dmatrix);

    drop(locked_model);

    // println!("Model predicted: {}", prediction);
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
        prediction: index_of_max.unwrap() as f32,
        accuracy: prediction_vec[index_of_max.unwrap()],
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env::set_var("RUST_BACKTRACE", "1");

    let model = match (
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
            println!("x[0] = {}", x_train[0]);

            let hidden_layer1 = Layer::new(relu, relu_derivative, x_train[0].len(), 512);

            let hidden_layer2 = Layer::new(relu, relu_derivative, 512, 512);

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
                128,
                15,
                0.001,
                metrics.clone(),
                &mut rmsprop,
                x_train,
                y_train,
            );

            println!("\nTest results:\n");

            model.test(metrics, &x_test, &y_test);

            Some(model)
        }
        _ => {
            println!("Error reading CSV file:");
            None
        }
    };

    let shared_data = Arc::new(Mutex::new(model.unwrap()));

    println!("Started API");

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
