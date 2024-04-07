use rand::distributions::Standard;
use rand::Rng;

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
    biases_hidden: Vec<f64>,
    biases_output: Vec<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights_input_hidden = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.sample::<f64, _>(Standard) - 0.5).collect())
            .collect();
        let weights_hidden_output = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.sample::<f64, _>(Standard) - 0.5).collect())
            .collect();
        let biases_hidden = vec![0.0; hidden_size];
        let biases_output = vec![0.0; output_size];

        Self {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            biases_hidden,
            biases_output,
        }
    }


    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        let hidden_layer_output = self.calculate_hidden_layer_output(input);
        self.calculate_output(&hidden_layer_output)
    }

    pub fn calculate_hidden_layer_output(&self, input: &[f64]) -> Vec<f64> {
        let mut hidden_layer_output = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = 0.0;
            for j in 0..self.input_size {
                sum += input[j] * self.weights_input_hidden[j][i];
            }
            hidden_layer_output[i] = sum + self.biases_hidden[i];
        }
        hidden_layer_output
    }

    pub fn calculate_output(&self, hidden_layer_output: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = 0.0;
            for j in 0..self.hidden_size {
                sum += hidden_layer_output[j] * self.weights_hidden_output[j][i];
            }
            output[i] = sum + self.biases_output[i];
        }
        output.iter().map(|&x| Self::sigmoid(x)).collect()
    }

    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for i in 0..inputs.len() {
                let input = &inputs[i];
                let target = &targets[i];
                let hidden_layer_output = self.calculate_hidden_layer_output(input);
                let output = self.calculate_output(&hidden_layer_output);
                let mut output_error = vec![0.0; self.output_size];
                for j in 0..self.output_size {
                    output_error[j] = (target[j] - output[j]) * output[j] * (1.0 - output[j]);
                }
                let mut hidden_error = vec![0.0; self.hidden_size];
                for j in 0..self.hidden_size {
                    let mut sum = 0.0;
                    for k in 0..self.output_size {
                        sum += output_error[k] * self.weights_hidden_output[j][k];
                    }
                    hidden_error[j] = sum * hidden_layer_output[j] * (1.0 - hidden_layer_output[j]);
                }
                for j in 0..self.hidden_size {
                    for k in 0..self.output_size {
                        self.weights_hidden_output[j][k] += learning_rate * output_error[k] * hidden_layer_output[j];
                    }
                }
                for j in 0..self.input_size {
                    for k in 0..self.hidden_size {
                        self.weights_input_hidden[j][k] += learning_rate * hidden_error[k] * input[j];
                    }
                }
                for j in 0..self.output_size {
                    self.biases_output[j] += learning_rate * output_error[j];
                }
                for j in 0..self.hidden_size {
                    self.biases_hidden[j] += learning_rate * hidden_error[j];
                }
            }
        }
    }
}



