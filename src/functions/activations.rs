use std::f64::consts::E;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
