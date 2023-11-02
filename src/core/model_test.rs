#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::{
        core::{layer::Layer, model::Model},
        functions::losses::{mse, mse_derivative},
    };

    #[test]
    fn test_evaluate_model() {
        let hidden_layer = Layer::from(
            |x| x,
            |x| x,
            DMatrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]),
            DMatrix::from_vec(2, 3, vec![0.5, 0.5, 0.1, 0.1, 0.7, 0.7]),
        );

        let output_layer = Layer::from(
            |x| x,
            |x| x,
            DMatrix::from_vec(1, 1, vec![1.0]),
            DMatrix::from_vec(3, 1, vec![0.0, 0.0, 0.0]),
        );

        let mut model = Model::new(vec![hidden_layer, output_layer], mse, mse_derivative);

        let data = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);

        // Output from hidden is Matrix1x3::new(1.5, 1.1, 1.7)

        assert_eq!(model.evaluate(&data), DVector::from_vec(vec![1.0]));
    }
}
