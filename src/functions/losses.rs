use nalgebra::DMatrix;

pub fn mse(expected: &DMatrix<f64>, predicted: &DMatrix<f64>) -> f64 {
    let n = expected.shape().1 as f64;

    let matrix = &(predicted - expected);
    let map = &matrix.map(|x| x.powf(2.0));
    let sum = map.sum();
    sum / n
}

pub fn mse_derivative(expected: &DMatrix<f64>, predicted: &DMatrix<f64>) -> DMatrix<f64> {
    predicted - expected
}

pub fn categorical_crossentropy(true_outputs: &DMatrix<f64>, pred_outputs: &DMatrix<f64>) -> f64 {
    let log_preds = pred_outputs.map(|pred| (pred + 1e-9).ln()); // Adding epsilon before logarithm
    let product = true_outputs.component_mul(&log_preds);
    let losses = product.row_sum();
    -losses.sum()
}

pub fn categorical_crossentropy_derivative(
    true_outputs: &DMatrix<f64>,
    pred_outputs: &DMatrix<f64>,
) -> DMatrix<f64> {
    let one_hot_predictions = pred_outputs.map(|x| if x == 0.0 { 1e-9 } else { x });

    -true_outputs.component_div(&one_hot_predictions)
}
