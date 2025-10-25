pub mod activator;

use crate::{
    dataset::Sample,
    model::Perceptron,
    neuralnetwork::activator::{Activator, ActivatorFn, Binary},
};

#[derive(Debug, Clone)]
pub struct PLAConfig {
    pub max_iteration: usize,
    pub fixed_learning_rate: f32,
    pub activator: ActivatorFn,
}

impl PLAConfig {
    pub const fn new() -> Self {
        Self {
            max_iteration: 1000,
            fixed_learning_rate: 0.001,
            activator: ActivatorFn::Binary(Binary::new()),
        }
    }
}

impl Default for PLAConfig {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PLResult {
    pub iteration: usize,
}

/// Perceptron Learning Algorithm
pub fn fit_perceptron<const N: usize>(
    config: &PLAConfig,
    perceptron: &mut Perceptron<N>,
    samples: &[Sample<N>],
) -> PLResult {
    let mut deltas = [0.0f32; N];

    for it in 1..config.max_iteration {
        for sample in samples {
            let sum = perceptron.solve(sample.inputs);
            let out = config.activator.activation(sum);
            let error = sample.activation - out;

            for (x, d) in sample.inputs.iter().copied().zip(deltas.iter_mut()) {
                *d += error * x;
            }
        }

        if deltas.iter().copied().all(|d| d == 0.0) {
            return PLResult { iteration: it };
        }

        for (d, w) in deltas.iter().copied().zip(perceptron.weights.iter_mut()) {
            let mean = d * config.fixed_learning_rate / samples.len() as f32;
            *w += mean;
        }

        deltas = [0.0; N];
    }

    PLResult {
        iteration: config.max_iteration,
    }
}
