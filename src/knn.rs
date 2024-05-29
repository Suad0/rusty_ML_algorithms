use std::collections::HashMap;

pub struct KNearestNeighbors {
    training_data: Vec<Vec<f64>>,
    training_labels: Vec<i32>,
    k: usize,
}

impl KNearestNeighbors {
    pub fn new(k: usize) -> Self {
        Self {
            training_data: Vec::new(),
            training_labels: Vec::new(),
            k,
        }
    }

    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<i32>) {
        self.training_data = data;
        self.training_labels = labels;
    }

    pub fn predict(&self, data: Vec<Vec<f64>>) -> Vec<i32> {
        data.iter().map(|point| self.predict_single(point)).collect()
    }

    fn predict_single(&self, point: &Vec<f64>) -> i32 {
        let distances: Vec<f64> = self.training_data
            .iter()
            .map(|training_point| self.euclidean_distance(training_point, point))
            .collect();

        let mut sorted_indices: Vec<usize> = (0..distances.len()).collect();
        sorted_indices.sort_by(|&i, &j| distances[i].partial_cmp(&distances[j]).unwrap());

        let k_nearest_labels: Vec<i32> = sorted_indices
            .into_iter()
            .take(self.k)
            .map(|index| self.training_labels[index])
            .collect();

        self.most_common_label(&k_nearest_labels)
    }

    fn euclidean_distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x1, x2)| (x1 - x2).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn most_common_label(&self, labels: &Vec<i32>) -> i32 {
        let mut frequency: HashMap<i32, i32> = HashMap::new();
        for &label in labels.iter() {
            *frequency.entry(label).or_insert(0) += 1;
        }
        *frequency.iter().max_by_key(|&(_, count)| count).map(|(label, _)| label).unwrap_or(&0)
    }
}
