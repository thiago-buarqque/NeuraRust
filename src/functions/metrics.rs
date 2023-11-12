use nalgebra::DMatrix;

#[derive(Debug, PartialEq, Clone)]
pub struct ConfusionMatrix {
    pub false_negatives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub true_positives: usize,
}

impl ConfusionMatrix {
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
) -> Vec<ConfusionMatrix> {
    let mut classes_confusion_matrices = Vec::new();

    for _ in 0..predictions[0].shape().1 {
        classes_confusion_matrices.push(ConfusionMatrix::empty())
    }

    for i in 0..predictions.len() {
        for j in 0..predictions[i].shape().1 {
            match (predictions[i][j].round() as u32, targets[i][j] as u32) {
                (1, 1) => classes_confusion_matrices[j].true_positives += 1,
                (0, 0) => classes_confusion_matrices[j].true_negatives += 1,
                (1, 0) => classes_confusion_matrices[j].false_positives += 1,
                (0, 1) => classes_confusion_matrices[j].false_negatives += 1,
                _ => {}
            }
        }
    }

    classes_confusion_matrices
}

pub fn print_metrics(
    epoch_predictions: Vec<DMatrix<f64>>,
    metrics: &Vec<String>,
    y: &Vec<DMatrix<f64>>,
) {
    let classes_confusion_matrices = calculate_confusion_matrix(&epoch_predictions, y);

    metrics.iter().for_each(|metric| {
        let mut scores: Vec<f64> = classes_confusion_matrices.iter().map(|_| 0.0_f64).collect();

        if metric.eq_ignore_ascii_case("accuracy") {
            scores = classes_confusion_matrices.iter().map(|cm| cm.accuracy()).collect();
        }

        if metric.eq_ignore_ascii_case("precision") {
            scores = classes_confusion_matrices.iter().map(|cm| cm.precision()).collect();
        }

        if metric.eq_ignore_ascii_case("recall") {
            scores = classes_confusion_matrices.iter().map(|cm| cm.recall()).collect();
        }

        if metric.eq_ignore_ascii_case("f1-score") {
            scores = classes_confusion_matrices.iter().map(|cm| cm.f1_score()).collect();
        }

        print!(
            " {}: {:.0}%",
            metric,(scores.iter().sum::<f64>() / classes_confusion_matrices.len() as f64) * 100.0
        );
    })
}