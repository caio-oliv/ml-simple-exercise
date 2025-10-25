pub mod preprocessing;

#[derive(Debug, Clone)]
pub struct Sample<const N: usize> {
    pub inputs: [f32; N],
    pub activation: f32,
}
