#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, Matrix3x1};

    use crate::core::layer::Layer;

    #[test]
    fn test_forward() {
        let mut layer = Layer::from(
            |x| x.clone(),
            |x| x.clone(),
            DMatrix::from_vec(3, 1, vec![1.0, 1.0, 1.0]),
            DMatrix::from_vec(3, 2, vec![0.5, 0.1, 0.7, 0.5, 0.1, 0.7]),
        );

        let data = DMatrix::from_vec(2, 1, vec![0.0, 1.0]);

        let r = layer.forward(&data).clone();

        assert_eq!(r, Matrix3x1::new(1.5, 1.1, 1.7))
    }
}
