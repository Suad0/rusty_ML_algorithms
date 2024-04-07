use crate::logistic_regression::LogisticRegressionModel;
use crate::neural_network::NeuralNetwork;

mod logistic_regression;
mod neural_network;


fn main() {
    // Sample training data
    let x_train = vec![
        vec![2.0, 1.0],
        vec![3.0, 2.0],
        vec![5.0, 4.0],
        vec![7.0, 5.0],
    ];
    let y_train = vec![0.0, 0.0, 1.0, 1.0]; // Sample labels

    // Hyperparameters
    let learning_rate = 0.01;
    let iterations = 1000;

    // Instantiate LogisticRegressionModel
    let mut model = LogisticRegressionModel::new();

    // Train the model
    model.train(x_train.clone(), &y_train, learning_rate, iterations);

    // Sample testing data
    let x_test = vec![
        vec![1.0, 1.0],
        vec![4.0, 3.0],
        vec![6.0, 5.0],
    ];

    // Make predictions
    let predictions = model.predict(x_test.clone());

    // Print predictions
    println!("Predictions:");
    for (i, pred) in predictions.iter().enumerate() {
        println!("Sample {}: {}", i + 1, pred);
    }



    let input_size = 2;
    let hidden_size = 3;
    let output_size = 1;

    // Create a new neural network
    let mut neural_network = NeuralNetwork::new(input_size, hidden_size, output_size);

    // Define sample inputs and targets
    let inputs = vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
        vec![0.7, 0.8],
    ];
    let targets = vec![
        vec![0.3],
        vec![0.5],
        vec![0.7],
        vec![0.9],
    ];

    // Train the neural network
    neural_network.train(&inputs, &targets, 1000, 0.1);

    // Test the trained neural network
    for input in &inputs {
        let prediction = neural_network.predict(input);
        println!("Input: {:?}, Prediction: {:?}", input, prediction);
    }


}

