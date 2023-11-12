#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::functions::metrics::{calculate_confusion_matrix, ConfusionMatrix};

    #[test]
    fn test_calculate_confusion_matrix() {
        let y_true = vec![
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
        ];

        let y_pred = vec![
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![0.0, 1.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
            DMatrix::from_vec(1, 2, vec![1.0, 0.0]),
        ];

        let confusion_matrix = calculate_confusion_matrix(&y_pred, &y_true);

        let expected = ConfusionMatrix {
            false_negatives: 1,
            false_positives: 1,
            true_negatives: 1,
            true_positives: 1,
        };
        assert_eq!(vec![expected.clone(), expected.clone()], confusion_matrix)
    }
}
