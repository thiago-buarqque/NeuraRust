use std::f32::consts::E;

use nalgebra::DMatrix;

pub fn sigmoid(raw_output: &DMatrix<f32>) -> DMatrix<f32> {
    raw_output.map(|x| 1.0 / (1.0 + E.powf(-x)))
}

pub fn sigmoid_derivative(raw_output: &DMatrix<f32>) -> DMatrix<f32> {
    raw_output.map(|x| {
        let s = 1.0 / (1.0 + E.powf(-x));
        s * (1.0 - s)
    })
}

pub fn softmax(input: &DMatrix<f32>) -> DMatrix<f32> {
    let exp_values = input.map(|x| E.powf(x));
    let sum_exp_values = exp_values.sum();

    exp_values / (sum_exp_values)
}

pub fn softmax_derivative(raw_output: &DMatrix<f32>) -> DMatrix<f32> {
    let n = raw_output.nrows();
    let m = raw_output.ncols();

    let mut jacobian = DMatrix::<f32>::from_element(n, m, 0.0);

    for i in 0..n {
        for j in 0..m {
            if i == j {
                jacobian[(i, j)] = raw_output[(i, j)] * (1.0 - raw_output[(i, j)]);
            } else {
                jacobian[(i, j)] = -raw_output[(i, j)] * raw_output[(i, j)];
            }
        }
    }

    jacobian
}

pub fn relu(raw_output: &DMatrix<f32>) -> DMatrix<f32> {
    raw_output.map(|x| if x > 0.0 { x } else { 0.0 })
}

pub fn relu_derivative(raw_output: &DMatrix<f32>) -> DMatrix<f32> {
    raw_output.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
}
