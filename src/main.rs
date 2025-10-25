mod assets;
mod dataset;
mod model;
mod neuralnetwork;

use crate::neuralnetwork::{NeuralNetwork, PLAConfig, activator::ActivatorFn, fit_neural_network};

fn main() {
    let mut network = NeuralNetwork::new(6, ActivatorFn::default());
    let config = PLAConfig::new();

    let samples = assets::get_samples();

    let result = fit_neural_network(&config, &mut network, &samples);

    println!("Finished on iteration: {}", result.iteration);
    println!("Perceptron weights: {:?}", network.perceptron.weights());

    assets::save_network("my_neural_network.json", &network).unwrap();
}
