use std::{
    fs,
    io::{self, BufReader, BufWriter},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    dataset::{self, Sample},
    neuralnetwork::NeuralNetwork,
};

pub const ASSETS_PATH: &str = "assets";

#[derive(Debug)]
pub enum SaveNetworkError {
    IO(io::Error),
    Serialization(serde_json::Error),
    Data,
    Unknown,
}

impl From<io::Error> for SaveNetworkError {
    fn from(error: io::Error) -> Self {
        Self::IO(error)
    }
}

impl From<serde_json::Error> for SaveNetworkError {
    fn from(error: serde_json::Error) -> Self {
        use serde_json::error::Category::*;
        if let Some(io_kind) = error.io_error_kind() {
            return Self::IO(io_kind.into());
        }

        match error.classify() {
            Data => Self::Data,
            Syntax => Self::Unknown,
            Eof => Self::Unknown,
            Io => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SignalData {
    pub signal_1: f32,
    pub signal_2: f32,
    pub signal_3: f32,
    pub signal_4: f32,
    pub signal_5: f32,
    pub signal_6: f32,
    pub action: f32,
}

impl SignalData {
    pub fn into_sample(self) -> Sample {
        let input = [
            self.signal_1,
            self.signal_2,
            self.signal_3,
            self.signal_4,
            self.signal_5,
            self.signal_6,
        ];
        Sample::new(&input, self.action)
    }
}

pub fn get_samples() -> Vec<Sample> {
    const SAMPLES_PATH: &str = "assets/samples.csv";

    let file = fs::OpenOptions::new()
        .read(true)
        .open(SAMPLES_PATH)
        .unwrap();
    let mut file_buf = BufReader::new(file);

    let samples: Vec<SignalData> = dataset::preprocessing::import_csv(&mut file_buf).unwrap();

    samples.into_iter().map(SignalData::into_sample).collect()
}

fn get_asset_path<P: AsRef<Path> + ?Sized>(file_name: &P) -> PathBuf {
    let mut path =
        PathBuf::with_capacity(ASSETS_PATH.len() + 1 + file_name.as_ref().as_os_str().len());
    path.push(ASSETS_PATH);
    path.push(file_name);
    path
}

pub fn save_network<P: AsRef<Path> + ?Sized>(
    file_name: &P,
    network: &NeuralNetwork,
) -> Result<(), SaveNetworkError> {
    let path = get_asset_path(file_name);

    let file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let mut writter = BufWriter::new(file);
    serde_json::to_writer(&mut writter, network)?;
    Ok(())
}
