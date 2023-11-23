#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::functions::losses::categorical_crossentropy;

    #[test]
    fn test_evaluate_model() {
        let y: DMatrix<f32> = DMatrix::from_vec(3, 1, vec![0.0, 0.0, 1.0]);
        let y_hat: DMatrix<f32> = DMatrix::from_vec(3, 1, vec![0.0, 0.7, 0.3]);

        assert_eq!(1.204, categorical_crossentropy(&y, &y_hat))
    }
}
