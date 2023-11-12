#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, Matrix1x3};

    use crate::core::layer::Layer;

    #[test]
    fn test_forward() {
        let mut layer = Layer::from(
            |x| x,
            |x| x,
            DMatrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]),
            DMatrix::from_vec(2, 3, vec![0.5, 0.5, 0.1, 0.1, 0.7, 0.7]),
        );

        let data = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);

        assert_eq!(layer.forward(&data).clone(), Matrix1x3::new(1.5, 1.1, 1.7))
    }
}
