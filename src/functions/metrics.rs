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

        let column_sum = confusion_matrix.column(class_index).sum();

        let false_positives = column_sum - true_positives;

        let row_sum = confusion_matrix.row(class_index).sum();

        let false_negatives = row_sum - true_positives;

        let true_negatives = confusion_matrix.sum() + true_positives - column_sum - row_sum;

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

    pub fn precision(&self) -> f32 {
        if self.true_positives + self.false_positives == 0 {
            0.0
        } else {
            self.true_positives as f32 / (self.true_positives + self.false_positives) as f32
        }
    }

    pub fn recall(&self) -> f32 {
        if self.true_positives + self.false_negatives == 0 {
            0.0
        } else {
            self.true_positives as f32 / (self.true_positives + self.false_negatives) as f32
        }
    }

    pub fn f1_score(&self) -> f32 {
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
    predictions: &Vec<DMatrix<f32>>,
    targets: &Vec<DMatrix<f32>>,
) -> DMatrix<usize> {
    let num_classes = predictions.first().unwrap().nrows();

    let mut confusion_matrix = DMatrix::zeros(num_classes, num_classes);

    for (act_matrix, exp_matrix) in predictions.iter().zip(targets.iter()) {
        let predicted_class = determine_predicted_class(act_matrix);
        let exp_class = determine_actual_class(exp_matrix);

        confusion_matrix[(exp_class, predicted_class)] += 1;
    }

    confusion_matrix
}

fn determine_predicted_class(matrix: &DMatrix<f32>) -> usize {
    matrix
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn determine_actual_class(matrix: &DMatrix<f32>) -> usize {
    matrix
        .iter()
        .enumerate()
        .find(|&(_, &value)| value == 1.0)
        .unwrap()
        .0
}

pub fn print_metrics(
    epoch_predictions: Vec<DMatrix<f32>>,
    metrics: &Vec<String>,
    y: &Vec<DMatrix<f32>>,
) {
    let confusion_matrix = calculate_confusion_matrix(&epoch_predictions, y);

    // println!("Confusion matrix: {}", confusion_matrix);

    let class_confusion_matrices = get_class_confusion_matrices(&confusion_matrix);

    metrics.iter().for_each(|metric| {
        let mut score: f32 = 0.0;

        if metric.eq_ignore_ascii_case("accuracy") {
            let total_correct_predictions = confusion_matrix.diagonal().sum();
            let total_predictions = confusion_matrix.sum();
            score = total_correct_predictions as f32 / total_predictions as f32;

            print!(" {}: {:.0}%", metric, (score) * 100.0);
        } else {
            if metric.eq_ignore_ascii_case("precision") {
                score = class_confusion_matrices
                    .iter()
                    .map(|cm| cm.precision())
                    .sum::<f32>();
            }

            if metric.eq_ignore_ascii_case("recall") {
                score = class_confusion_matrices
                    .iter()
                    .map(|cm| cm.recall())
                    .sum::<f32>();
            }

            if metric.eq_ignore_ascii_case("f1-score") {
                score = class_confusion_matrices
                    .iter()
                    .map(|cm| cm.f1_score())
                    .sum::<f32>();
            }

            print!(
                " {}: {:.0}%",
                metric,
                (score / class_confusion_matrices.len() as f32) * 100.0
            );
        }
    })
}

pub fn get_class_confusion_matrices(
    confusion_matrix: &DMatrix<usize>,
) -> Vec<ClassConfusionMatrix> {
    let mut class_confusion_matrices = Vec::new();

    for class_index in 0..confusion_matrix.nrows() {
        class_confusion_matrices.push(ClassConfusionMatrix::new(confusion_matrix, class_index));
    }

    class_confusion_matrices
}
