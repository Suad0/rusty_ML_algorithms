pub struct LogisticRegressionModel {
    coefficients: Vec<Vec<f64>>,
}

impl LogisticRegressionModel {
    pub fn new() -> Self {
        LogisticRegressionModel {
            coefficients: vec![],
        }
    }

    pub fn train(
        &mut self,
        x_train: &[Vec<f64>],
        y_train: &[f64],
        learning_rate: f64,
        iterations: usize,
    ) {
        let num_features = x_train[0].len();
        self.coefficients = vec![vec![0.0; num_features + 1]; 1]; // Include bias term

        // Add bias term to features (append 1 to each row of X_train)
        let mut augmented_x_train = vec![vec![0.0; num_features + 1]; x_train.len()];
        for i in 0..x_train.len() {
            augmented_x_train[i][0] = 1.0; // Bias term
            augmented_x_train[i][1..].copy_from_slice(&x_train[i]);
        }

        // Gradient Descent
        for _ in 0..iterations {
            let predictions = self.predict_probability(&augmented_x_train);

            let mut errors = vec![0.0; y_train.len()];
            for i in 0..y_train.len() {
                errors[i] = predictions[i] - y_train[i];
            }

            for j in 0..=num_features {
                let mut gradient = 0.0;
                for i in 0..x_train.len() {
                    gradient += errors[i] * augmented_x_train[i][j];
                }
                self.coefficients[0][j] -= learning_rate * gradient / x_train.len() as f64;
            }
        }
    }

    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn predict_probability(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut predictions = vec![0.0; x.len()];
        for i in 0..x.len() {
            let mut linear_combination = 0.0;
            for j in 0..x[0].len() {
                linear_combination += self.coefficients[0][j] * x[i][j];
            }
            predictions[i] = self.sigmoid(linear_combination);
        }
        predictions
    }

    pub fn predict(&self, x_test: &[Vec<f64>]) -> Vec<f64> {
        let mut augmented_x_test = vec![vec![0.0; x_test[0].len() + 1]; x_test.len()];
        for i in 0..x_test.len() {
            augmented_x_test[i][0] = 1.0; // Bias term
            augmented_x_test[i][1..].copy_from_slice(&x_test[i]);
        }
        self.predict_probability(&augmented_x_test)
    }
}

fn main() {
    //  data for training
    let x_train = vec![
        vec![2.0, 3.0],
        vec![1.0, 2.0],
        vec![5.0, 3.0],
        vec![2.0, 4.0],
    ];
    let y_train = vec![0.0, 1.0, 0.0, 1.0];

    let mut model = LogisticRegressionModel::new();

    model.train(&x_train, &y_train, 0.01, 100);

    // Sample data for testing
    let x_test = vec![
        vec![3.0, 4.0],
        vec![1.0, 1.0],
    ];

    // Make predictions
    let predictions = model.predict(&x_test);

    // Display predictions
    println!("Predictions:");
    for (i, prediction) in predictions.iter().enumerate() {
        println!("Sample {}: {}", i + 1, prediction);
    }
}
