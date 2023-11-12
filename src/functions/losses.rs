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
    assert_eq!(true_outputs.ncols(), pred_outputs.ncols());
    assert_eq!(true_outputs.nrows(), pred_outputs.nrows());

    let log_preds = pred_outputs.map(|pred| if pred <= 0.0 { 0.0 } else { pred.ln() });
    let product = true_outputs.component_mul(&log_preds);
    let losses = product.row_sum();
    -losses.sum()
}

pub fn categorical_crossentropy_derivative(true_outputs: &DMatrix<f64>, pred_outputs: &DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(true_outputs.ncols(), pred_outputs.ncols());
    assert_eq!(true_outputs.nrows(), pred_outputs.nrows());

    pred_outputs - true_outputs
}