#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::{
        core::{layer::Layer, model::Model},
        functions::{
            activations::{sigmoid, sigmoid_derivative},
            losses::{mse, mse_derivative, squared_error, squared_error_derivative},
        },
        optimizers::rmsprop::RMSProp,
    };

    #[test]
    fn test_evaluate_model() {
        let hidden_layer = Layer::from(
            |x| x.clone(),
            |x| x.clone(),
            DMatrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]),
            DMatrix::from_vec(2, 3, vec![0.5, 0.5, 0.1, 0.1, 0.7, 0.7]),
        );

        let output_layer = Layer::from(
            |x| x.clone(),
            |x| x.clone(),
            DMatrix::from_vec(1, 1, vec![1.0]),
            DMatrix::from_vec(3, 1, vec![0.0, 0.0, 0.0]),
        );

        let mut model = Model::new(vec![hidden_layer, output_layer], mse, mse_derivative);

        let data = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);

        // Output from hidden is Matrix1x3::new(1.5, 1.1, 1.7)

        assert_eq!(model.evaluate(&data), DVector::from_vec(vec![1.0]));
    }

    #[test]
    fn test_evaluate_model_2() {
        let hidden_layer = Layer::from(
            sigmoid,
            sigmoid_derivative,
            DMatrix::from_vec(2, 1, vec![0.35, 0.35]),
            DMatrix::from_vec(2, 2, vec![0.15, 0.20, 0.25, 0.30]),
        );

        let output_layer = Layer::from(
            sigmoid,
            sigmoid_derivative,
            DMatrix::from_vec(2, 1, vec![0.60, 0.60]),
            DMatrix::from_vec(2, 2, vec![0.40, 0.45, 0.50, 0.55]),
        );

        let mut model = Model::new(
            vec![hidden_layer, output_layer],
            squared_error,
            squared_error_derivative,
        );

        let data = DMatrix::from_vec(2, 1, vec![0.05, 0.10]);

        // Output from hidden is Matrix1x3::new(1.5, 1.1, 1.7)

        let mut rmsprop = RMSProp::new(0.9);

        model.fit(
            1,
            1,
            0.5,
            vec![],
            &mut rmsprop,
            vec![data.clone()],
            vec![DMatrix::from_vec(2, 1, vec![0.01, 0.99])],
        );

        let output = model.evaluate(&data);

        assert_eq!(
            output,
            DVector::from_vec(vec![0.75136507, 0.772928465])
        );
    }
}
