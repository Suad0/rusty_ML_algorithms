pub struct LogisticRegressionModel {
    coefficients: Vec<Vec<f64>>,
}

impl LogisticRegressionModel {
    pub fn new() -> Self {
        LogisticRegressionModel {
            coefficients: vec![],
        }
    }

    pub fn train(&mut self, x_train: &[Vec<f64>], y_train: &[f64], learning_rate: f64, iterations: usize) {
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

use std::collections::HashMap;

struct DecisionTree {
    root: Option<Box<Node>>,
    max_depth: usize,
}

impl DecisionTree {
    fn new(max_depth: usize) -> Self {
        DecisionTree {
            root: None,
            max_depth,
        }
    }

    fn train(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<usize>) {
        self.root = Some(Box::new(self.build_tree(&x_train, &y_train, 0)));
    }

    fn predict(&self, instance: &[f64]) -> usize {
        self.classify(instance, self.root.as_ref().unwrap())
    }

    fn build_tree(&self, x_train: &[Vec<f64>], y_train: &[usize], depth: usize) -> Node {
        if depth >= self.max_depth || self.all_same(y_train) {
            return Node::new(self.most_common_class(y_train));
        }

        let best_split = self.find_best_split(x_train, y_train);

        if best_split.is_none() {
            return Node::new(self.most_common_class(y_train));
        }

        let (split, left_x, left_y, right_x, right_y) = best_split.unwrap();

        let left_child = Box::new(self.build_tree(&left_x, &left_y, depth + 1));
        let right_child = Box::new(self.build_tree(&right_x, &right_y, depth + 1));

        Node::with_children(
            split.feature_index,
            split.threshold,
            left_child,
            right_child,
        )
    }

    fn all_same(&self, array: &[usize]) -> bool {
        let first = array[0];
        array.iter().all(|&x| x == first)
    }

    fn most_common_class(&self, array: &[usize]) -> usize {
        let mut class_counts = HashMap::new();
        for &value in array {
            *class_counts.entry(value).or_insert(0) += 1;
        }
        *class_counts.iter().max_by_key(|&(_, count)| count).unwrap().0
    }


    fn find_best_split(
        &self,
        x_train: &[Vec<f64>],
        y_train: &[usize],
    ) -> Option<(Split, Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>)> {
        let num_features = x_train[0].len();
        let num_rows = x_train.len();
        let mut best_gini = f64::MAX;
        let mut best_split: Option<(Split, Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>)> = None;

        for feature_index in 0..num_features {
            for row_index in 0..num_rows {
                let threshold = x_train[row_index][feature_index];

                let mut left_y = Vec::new();
                let mut left_x = Vec::new();
                let mut right_y = Vec::new();
                let mut right_x = Vec::new();

                for i in 0..num_rows {
                    if x_train[i][feature_index] <= threshold {
                        left_y.push(y_train[i]);
                        left_x.push(x_train[i].clone());
                    } else {
                        right_y.push(y_train[i]);
                        right_x.push(x_train[i].clone());
                    }
                }

                let gini = self.calculate_gini(&[&left_y, &right_y]);
                if gini < best_gini {
                    best_gini = gini;
                    let split = Split::new(feature_index, threshold);
                    best_split = Some((split, left_x, left_y, right_x, right_y));
                }
            }
        }

        best_split
    }

    fn calculate_gini(&self, groups: &[&[usize]]) -> f64 {
        let total_instances: usize = groups.iter().map(|group| group.len()).sum();

        let mut gini = 0.0;
        for group in groups {
            let group_size = group.len() as f64;
            if group_size == 0.0 {
                continue;
            }
            let mut score = 0.0;
            for &class_label in *group {
                let p = group.iter().filter(|&&x| x == class_label).count() as f64 / group_size;
                score += p * p;
            }
            gini += (1.0 - score) * (group_size / total_instances as f64);
        }
        gini
    }


    fn classify(&self, instance: &[f64], node: &Node) -> usize {
        if node.is_leaf() {
            return node.class_label;
        }

        if instance[node.feature_index] <= node.threshold {
            self.classify(instance, &*node.left_child.as_ref().unwrap())
        } else {
            self.classify(instance, &*node.right_child.as_ref().unwrap())
        }
    }
}

struct Node {
    feature_index: usize,
    threshold: f64,
    class_label: usize,
    left_child: Option<Box<Node>>,
    right_child: Option<Box<Node>>,
}

impl Node {
    fn new(class_label: usize) -> Self {
        Node {
            feature_index: 0,
            threshold: 0.0,
            class_label,
            left_child: None,
            right_child: None,
        }
    }

    fn with_children(
        feature_index: usize,
        threshold: f64,
        left_child: Box<Node>,
        right_child: Box<Node>,
    ) -> Self {
        Node {
            feature_index,
            threshold,
            class_label: 0,
            left_child: Some(left_child),
            right_child: Some(right_child),
        }
    }

    fn is_leaf(&self) -> bool {
        self.left_child.is_none() && self.right_child.is_none()
    }
}

struct Split {
    feature_index: usize,
    threshold: f64,
}

impl Split {
    fn new(feature_index: usize, threshold: f64) -> Self {
        Split {
            feature_index,
            threshold,
        }
    }
}

fn main() {
    let x_train = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![4.0, 5.0],
        vec![5.0, 6.0],
        vec![6.0, 7.0],
        vec![7.0, 8.0],
        vec![8.0, 9.0],
    ];

    let y_train = vec![0, 0, 1, 1, 0, 1, 0, 1];

    let mut decision_tree = DecisionTree::new(3);
    decision_tree.train(x_train, y_train);

    let x_test = vec![
        vec![1.0, 2.0],
        vec![5.0, 6.0],
        vec![8.0, 9.0],
    ];

    for instance in &x_test {
        let prediction = decision_tree.predict(instance);
        println!("Prediction for {:?}: {}", instance, prediction);
    }
}

