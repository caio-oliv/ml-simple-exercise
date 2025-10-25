pub mod preprocessing;

#[derive(Debug, Clone)]
pub struct Sample {
    inputs: Box<[f32]>,
    activation: f32,
}

impl Sample {
    pub fn new(inputs: &[f32], activation: f32) -> Self {
        Self::from_iter(inputs.iter().copied(), activation)
    }

    pub fn from_iter<I>(inputs: I, activation: f32) -> Self
    where
        I: Iterator<Item = f32>,
    {
        Self {
            inputs: inputs.collect(),
            activation,
        }
    }

    pub const fn inputs(&self) -> &[f32] {
        &self.inputs
    }

    pub const fn activation(&self) -> f32 {
        self.activation
    }
}
