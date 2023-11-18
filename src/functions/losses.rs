use nalgebra::DMatrix;

pub fn mse(expected: &DMatrix<f32>, predicted: &DMatrix<f32>) -> f32 {
    let n = expected.shape().1 as f32;

    let matrix = &(predicted - expected);
    let map = &matrix.map(|x| x.powf(2.0));
    let sum = map.sum();
    sum / n
}

pub fn mse_derivative(expected: &DMatrix<f32>, predicted: &DMatrix<f32>) -> DMatrix<f32> {
    predicted - expected
}

// Function to calculate the squared error
pub fn squared_error(expected: &DMatrix<f32>, predicted: &DMatrix<f32>) -> f32 {
    (expected - predicted).map(|x| x.powf(2.0) / 2.0).sum()
}

// Function to calculate the derivative of the squared error
pub fn squared_error_derivative(expected: &DMatrix<f32>, predicted: &DMatrix<f32>) -> DMatrix<f32> {
    -(expected - predicted)
}

pub fn categorical_crossentropy(expected: &DMatrix<f32>, predicted: &DMatrix<f32>) -> f32 {
    let log_preds = predicted.map(|pred| (pred + 1e-15).ln());
    let product = expected.component_mul(&log_preds);

    -product.sum()
}

pub fn categorical_crossentropy_derivative(
    expected: &DMatrix<f32>,
    predicted: &DMatrix<f32>,
) -> DMatrix<f32> {
    -expected.component_div(&predicted.map(|x| if x == 0.0 { 1e-15 } else { x }))
}
