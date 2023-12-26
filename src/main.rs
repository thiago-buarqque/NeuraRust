#![allow(warnings)]
use std::env;
use std::sync::{Arc, Mutex};

use nalgebra::DMatrix;

use crate::core::model::Model;
use crate::model_handler::get_trained_model;

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};

mod core;
mod functions;
mod model_handler;
mod optimizers;

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

    let input_data = DMatrix::from_row_slice(matrix_elements.len(), 1, &matrix_elements);
    let mut locked_model = shared.lock().unwrap();

    let prediction = locked_model.evaluate(&input_data);

    drop(locked_model);

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

    println!("Training the network...");

    let model = get_trained_model();

    let shared_data = Arc::new(Mutex::new(model.unwrap()));

    println!("\n===== Started API =====");

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
