use core::f32;

pub trait Activator {
    fn activation(&self, x: f32) -> f32;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActivatorFn {
    Binary(Binary),
    Tanh(HyperbolicTangent),
    Logistic(Logistic),
}

impl ActivatorFn {
    pub const fn activator(&self) -> &dyn Activator {
        match self {
            Self::Binary(bin) => bin,
            Self::Tanh(tanh) => tanh,
            Self::Logistic(logis) => logis,
        }
    }
}

impl Default for ActivatorFn {
    fn default() -> Self {
        Self::Binary(Binary::default())
    }
}

impl Activator for ActivatorFn {
    fn activation(&self, x: f32) -> f32 {
        match self {
            Self::Binary(bin) => bin.activation(x),
            Self::Tanh(tanh) => tanh.activation(x),
            Self::Logistic(logis) => logis.activation(x),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binary {
    pub threshold: f32,
    pub low: f32,
    pub high: f32,
}

impl Binary {
    pub const fn new() -> Self {
        Self {
            threshold: 0.0,
            low: -1.0,
            high: 1.0,
        }
    }
}

impl Default for Binary {
    fn default() -> Self {
        Self::new()
    }
}

impl Activator for Binary {
    fn activation(&self, x: f32) -> f32 {
        if x < self.threshold {
            self.low
        } else {
            self.high
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperbolicTangent;

impl Activator for HyperbolicTangent {
    fn activation(&self, x: f32) -> f32 {
        f32::tanh(x)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Logistic;

impl Activator for Logistic {
    fn activation(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gaussian {
    pub mean: f32,
    pub std_dev: f32,
}

fn gaussian(x: f32, mean: f32, std_dev: f32) -> f32 {
    let exponent = -((x - mean).powi(2)) / (2.0 * std_dev.powi(2));
    (1.0 / (std_dev * (2.0 * f32::consts::PI).sqrt())) * exponent.exp()
}

impl Gaussian {
    pub const fn new() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self::new()
    }
}

impl Activator for Gaussian {
    fn activation(&self, x: f32) -> f32 {
        gaussian(x, self.mean, self.std_dev)
    }
}
