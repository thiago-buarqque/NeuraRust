#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::functions::metrics::{
        calculate_confusion_matrix, get_class_confusion_matrices,
    };

    #[test]
    fn test_calculate_confusion_matrix() {
        let y_pred = vec![
            DMatrix::from_vec(3, 1, vec![0.05, 0.95, 0.2]),
            DMatrix::from_vec(3, 1, vec![0.73, 0.55, 0.1]),
            DMatrix::from_vec(3, 1, vec![0.1, 0.5, 0.95]),
            DMatrix::from_vec(3, 1, vec![0.2, 0.4, 0.85]),
        ];

        let y_true = vec![
            DMatrix::from_vec(3, 1, vec![0.0, 1.0, 0.0]),
            DMatrix::from_vec(3, 1, vec![1.0, 0.0, 0.0]),
            DMatrix::from_vec(3, 1, vec![0.0, 0.0, 1.0]),
            DMatrix::from_vec(3, 1, vec![0.0, 1.0, 0.0]),
        ];

        let confusion_matrix = calculate_confusion_matrix(&y_pred, &y_true);

        assert_eq!(
            DMatrix::from_vec(3, 3, vec![1, 0, 0, 0, 1, 0, 0, 1, 1]),
            confusion_matrix
        );
    }

    #[test]
    fn test_metrics() {
        let confusion_matrix = DMatrix::from_vec(3, 3, vec![1, 0, 0, 0, 1, 0, 0, 1, 1]);

        let class_confusion_matrices = get_class_confusion_matrices(&confusion_matrix);

        assert_eq!(1.0, class_confusion_matrices[0].precision());
        assert_eq!(1.0, class_confusion_matrices[1].precision());
        assert_eq!(0.5, class_confusion_matrices[2].precision());

        assert_eq!(1.0, class_confusion_matrices[0].recall());
        assert_eq!(0.5, class_confusion_matrices[1].recall());
        assert_eq!(1.0, class_confusion_matrices[2].recall());

        assert_eq!(1.0, class_confusion_matrices[0].f1_score());
        assert_eq!(0.6666666666666666, class_confusion_matrices[1].f1_score());
        assert_eq!(0.6666666666666666, class_confusion_matrices[2].f1_score());

        let total_correct_predictions = confusion_matrix.diagonal().sum();
        let total_predictions = confusion_matrix.sum();
        let overall_accuracy = total_correct_predictions as f32 / total_predictions as f32;

        assert_eq!(0.75, overall_accuracy);

        assert_eq!(
            0.8333333333333334,
            class_confusion_matrices
                .iter()
                .map(|cm| cm.precision())
                .sum::<f32>()
                / class_confusion_matrices.len() as f32
        );

        assert_eq!(
            0.8333333333333334,
            class_confusion_matrices
                .iter()
                .map(|cm| cm.recall())
                .sum::<f32>()
                / class_confusion_matrices.len() as f32
        );

        assert_eq!(
            0.77777785,
            class_confusion_matrices
                .iter()
                .map(|cm| cm.f1_score())
                .sum::<f32>()
                / class_confusion_matrices.len() as f32
        );
    }
}
