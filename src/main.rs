mod dataset;
mod model;
mod neuralnetwork;

use std::{fs, io::BufReader};

use serde::Deserialize;

use crate::{
    dataset::Sample,
    model::Perceptron,
    neuralnetwork::{PLAConfig, fit_perceptron},
};

const ASSETS_PATH: &str = "assets";
const SAMPLES_PATH: &str = "assets/samples.csv";

fn main() {
    let mut perceptron = Perceptron::<6>::new();
    let config = PLAConfig::new();

    let samples = get_samples();

    let result = fit_perceptron(&config, &mut perceptron, &samples);

    println!("Finished on iteration: {}", result.iteration);
    println!("Perceptron weights: {:?}", &perceptron.weights);
}

#[derive(Debug, Clone, Deserialize)]
struct SignalData {
    pub signal_1: f32,
    pub signal_2: f32,
    pub signal_3: f32,
    pub signal_4: f32,
    pub signal_5: f32,
    pub signal_6: f32,
    pub action: f32,
}

impl SignalData {
    pub const fn into_sample(self) -> Sample<6> {
        Sample {
            inputs: [
                self.signal_1,
                self.signal_2,
                self.signal_3,
                self.signal_4,
                self.signal_5,
                self.signal_6,
            ],
            activation: self.action,
        }
    }
}

fn get_samples() -> Vec<Sample<6>> {
    let file = fs::OpenOptions::new()
        .read(true)
        .open(SAMPLES_PATH)
        .unwrap();
    let mut file_buf = BufReader::new(file);

    let samples: Vec<SignalData> = dataset::preprocessing::import_csv(&mut file_buf).unwrap();

    samples.into_iter().map(SignalData::into_sample).collect()
}
