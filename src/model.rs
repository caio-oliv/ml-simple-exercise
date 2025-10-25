use core::mem::MaybeUninit;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perceptron {
    weights: Box<[f32]>,
    bias: f32,
}

impl Perceptron {
    pub fn new(len: usize) -> Self {
        let mut weights: Box<[MaybeUninit<f32>]> = Box::new_uninit_slice(len);
        weights.fill(MaybeUninit::new(1.0));

        Self {
            weights: unsafe { weights.assume_init() },
            bias: -1.0,
        }
    }

    pub const fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [f32] {
        self.weights.as_mut()
    }

    pub const fn bias(&self) -> f32 {
        self.bias
    }

    pub const fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn solve(&self, inputs: &[f32]) -> f32 {
        self.weights
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias
    }
}
