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