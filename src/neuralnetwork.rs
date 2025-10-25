pub mod activator;

use core::iter;

use serde::{Deserialize, Serialize};

use crate::{
    dataset::Sample,
    model::Perceptron,
    neuralnetwork::activator::{Activator, ActivatorFn},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub perceptron: Perceptron,
    pub activator: ActivatorFn,
}

impl NeuralNetwork {
    pub fn new(len: usize, activator: ActivatorFn) -> Self {
        Self {
            perceptron: Perceptron::new(len),
            activator,
        }
    }
}

/// Perceptron Learning Algorithm Configuration
#[derive(Debug, Clone)]
pub struct PLAConfig {
    pub max_iteration: usize,
    pub fixed_learning_rate: f32,
}

impl PLAConfig {
    pub const fn new() -> Self {
        Self {
            max_iteration: 1000,
            fixed_learning_rate: 0.001,
        }
    }
}

impl Default for PLAConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitResult {
    pub iteration: usize,
}

/// Perceptron Learning Algorithm
pub fn fit_neural_network(
    config: &PLAConfig,
    nnetwork: &mut NeuralNetwork,
    samples: &[Sample],
) -> FitResult {
    let mut deltas: Box<[f32]> = iter::repeat_n(0.0f32, nnetwork.perceptron.len()).collect();

    for it in 1..config.max_iteration {
        for sample in samples {
            let sum = nnetwork.perceptron.solve(sample.inputs());
            let out = nnetwork.activator.activation(sum);
            let error = sample.activation() - out;

            for (x, d) in sample.inputs().iter().copied().zip(deltas.iter_mut()) {
                *d += error * x;
            }
        }

        if deltas.iter().copied().all(|d| d == 0.0) {
            return FitResult { iteration: it };
        }

        for (d, w) in deltas
            .iter()
            .copied()
            .zip(nnetwork.perceptron.weights_mut().iter_mut())
        {
            let mean = d * config.fixed_learning_rate / samples.len() as f32;
            *w += mean;
        }

        deltas.fill(0.0);
    }

    FitResult {
        iteration: config.max_iteration,
    }
}
