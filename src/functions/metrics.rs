use nalgebra::DMatrix;

#[derive(Debug, PartialEq, Clone)]
pub struct ClassConfusionMatrix {
    pub false_negatives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub true_positives: usize,
}

impl ClassConfusionMatrix {
    pub fn new(confusion_matrix: &DMatrix<usize>, class_index: usize) -> Self {
        let true_positives = confusion_matrix[(class_index, class_index)];
        let false_positives = confusion_matrix.column(class_index).sum() - true_positives;
        let false_negatives = confusion_matrix.row(class_index).sum() - true_positives;
        // let true_negatives =
        //     confusion_matrix.sum() - false_positives - false_negatives - true_positives;
        let true_negatives = confusion_matrix.sum()
            - confusion_matrix.column(class_index).sum()
            - confusion_matrix.row(class_index).sum()
            + true_positives;

        ClassConfusionMatrix {
            false_negatives,
            false_positives,
            true_negatives,
            true_positives,
        }
    }

    pub fn empty() -> Self {
        Self {
            false_negatives: 0,
            false_positives: 0,
            true_negatives: 0,
            true_positives: 0,
        }
    }

    pub fn accuracy(&self) -> f64 {
        (self.true_positives + self.true_negatives) as f64
            / (self.true_positives
                + self.true_negatives
                + self.false_positives
                + self.false_negatives) as f64
    }

    pub fn precision(&self) -> f64 {
        if self.true_positives + self.false_positives == 0 {
            0.0
        } else {
            self.true_positives as f64 / (self.true_positives + self.false_positives) as f64
        }
    }

    pub fn recall(&self) -> f64 {
        if self.true_positives + self.false_negatives == 0 {
            0.0
        } else {
            self.true_positives as f64 / (self.true_positives + self.false_negatives) as f64
        }
    }

    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            false_negatives: self.false_negatives,
            false_positives: self.false_positives,
            true_negatives: self.true_negatives,
            true_positives: self.true_positives,
        }
    }
}

pub fn calculate_confusion_matrix(
    predictions: &Vec<DMatrix<f64>>,
    targets: &Vec<DMatrix<f64>>,
) -> DMatrix<usize> {
    let num_classes = predictions.first().unwrap().ncols(); // Assuming all matrices have the same number of columns

    // Initialize confusion matrix
    let mut confusion_matrix = DMatrix::zeros(num_classes, num_classes);

    // Populate confusion matrix
    for (act_matrix, exp_matrix) in predictions.iter().zip(targets.iter()) {
        let predicted_class = determine_predicted_class(act_matrix);
        let actual_class = determine_actual_class(exp_matrix);
        confusion_matrix[(actual_class, predicted_class)] += 1;
    }

    confusion_matrix
}

fn determine_predicted_class(matrix: &DMatrix<f64>) -> usize {
    matrix
        .row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn determine_actual_class(matrix: &DMatrix<f64>) -> usize {
    matrix
        .row(0)
        .iter()
        .enumerate()
        .find(|&(_, &value)| value == 1.0)
        .unwrap()
        .0
}

pub fn print_metrics(
    epoch_predictions: Vec<DMatrix<f64>>,
    metrics: &Vec<String>,
    y: &Vec<DMatrix<f64>>,
) {
    let confusion_matrix = calculate_confusion_matrix(&epoch_predictions, y);

    let class_confusion_matrices = get_class_confusion_matrices(&confusion_matrix);

    metrics.iter().for_each(|metric| {
        let mut score: f64 = 0.0;

        if metric.eq_ignore_ascii_case("accuracy") {
            let total_correct_predictions = confusion_matrix.diagonal().sum();
            let total_predictions = confusion_matrix.sum();
            score = total_correct_predictions as f64 / total_predictions as f64;
        }

        if metric.eq_ignore_ascii_case("precision") {
            score = class_confusion_matrices
                .iter()
                .map(|cm| cm.precision())
                .sum::<f64>();
        }

        if metric.eq_ignore_ascii_case("recall") {
            score = class_confusion_matrices
                .iter()
                .map(|cm| cm.recall())
                .sum::<f64>();
        }

        if metric.eq_ignore_ascii_case("f1-score") {
            score = class_confusion_matrices
                .iter()
                .map(|cm| cm.f1_score())
                .sum::<f64>();
        }

        print!(
            " {}: {:.0}%",
            metric,
            (score / class_confusion_matrices.len() as f64) * 100.0
        );
    })
}

pub fn get_class_confusion_matrices(
    confusion_matrix: &DMatrix<usize>,
) -> Vec<ClassConfusionMatrix> {
    let num_classes = confusion_matrix.nrows();
    let mut class_confusion_matrices = Vec::new();

    for class_index in 0..num_classes {
        let class_cm = ClassConfusionMatrix::new(confusion_matrix, class_index);
        class_confusion_matrices.push(class_cm);
    }

    class_confusion_matrices
}
