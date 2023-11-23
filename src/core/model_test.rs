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
            DMatrix::from_vec(3, 1, vec![1.0, 1.0, 1.0]),
            DMatrix::from_vec(3, 2, vec![0.5, 0.1, 0.7, 0.5, 0.1, 0.7]),
        );

        let output_layer = Layer::from(
            |x| x.clone(),
            |x| x.clone(),
            DMatrix::from_vec(1, 1, vec![1.0]),
            DMatrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]),
        );

        let mut model = Model::new(vec![hidden_layer, output_layer], mse, mse_derivative);

        let data = DMatrix::from_vec(2, 1, vec![0.0, 1.0]);

        assert_eq!(model.evaluate(&data), DVector::from_vec(vec![1.0]));
    }

}
