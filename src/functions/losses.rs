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
    expected - predicted
    // -expected.component_div(&predicted.map(|x| if x == 0.0 { 1e-15 } else { x }))
}

pub fn binary_crossentropy(y_true: &DMatrix<f32>, y_pred: &DMatrix<f32>) -> f32 {
    if y_true.shape() != y_pred.shape() {
        panic!("Shapes of y_true and y_pred must match");
    }

    let epsilon = 1e-7; // To prevent log(0)
    let mut loss = 0.0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let yp_clamped = yp.max(epsilon).min(1.0 - epsilon); // Clamping values for stability
        loss += -yt * yp_clamped.ln() - (1.0 - yt) * (1.0 - yp_clamped).ln();
    }

    loss / (y_true.nrows() * y_true.ncols()) as f32
}

pub fn binary_crossentropy_derivative(
    expected: &DMatrix<f32>,
    predicted: &DMatrix<f32>,
) -> DMatrix<f32> {
    let mut derivatives = DMatrix::zeros(expected.nrows(), expected.ncols());

    for ((y, y_hat), derivative) in expected
        .iter()
        .zip(predicted.iter())
        .zip(derivatives.iter_mut())
    {
        *derivative = -(*y / *y_hat) + ((1.0 - *y) / (1.0 - *y_hat));
    }

    derivatives
}
