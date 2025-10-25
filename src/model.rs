pub struct Perceptron<const N: usize> {
    pub weights: [f32; N],
    pub bias: f32,
}

impl<const N: usize> Perceptron<N> {
    pub const fn new() -> Self {
        Self {
            weights: [1.0; N],
            bias: -1.0,
        }
    }

    pub fn solve(&self, inputs: [f32; N]) -> f32 {
        self.weights
            .iter()
            .copied()
            .zip(inputs)
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias
    }
}
