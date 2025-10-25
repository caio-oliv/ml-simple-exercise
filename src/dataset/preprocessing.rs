use core::{
    error,
    fmt::{self, Display},
};
use std::io;

use csv::{DeserializeError, Position, Reader, ReaderBuilder, StringRecord};
use serde::de::DeserializeOwned;

#[derive(Debug)]
pub enum ImportCsvError {
    IO(io::Error),
    Utf8 {
        position: Option<Position>,
        error: csv::Utf8Error,
    },
    UnreadableHeader,
    RecordLength {
        position: Option<Position>,
        expected_len: u64,
        len: u64,
    },
    Deserialize {
        position: Option<Position>,
        error: DeserializeError,
    },
    Unknown,
}

fn fmt_variant_message(
    f: &mut fmt::Formatter<'_>,
    message: &str,
    position: Option<&Position>,
) -> fmt::Result {
    f.write_str(message)?;
    if let Some(pos) = position {
        f.write_str(" at line ")?;
        pos.line().fmt(f)?;
    }
    Ok(())
}

fn fmt_variant_message_with_error<E>(
    f: &mut fmt::Formatter<'_>,
    message: &str,
    position: Option<&Position>,
    error: &E,
) -> fmt::Result
where
    E: error::Error,
{
    fmt_variant_message(f, message, position)?;
    f.write_str(": ")?;
    fmt::Display::fmt(&error, f)?;
    Ok(())
}

impl fmt::Display for ImportCsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Import CSV ")?;

        match self {
            Self::IO(error) => {
                f.write_str("IO error: ")?;
                error.fmt(f)?;
            }
            Self::Utf8 { position, error } => {
                fmt_variant_message_with_error(f, "UTF8 parsing error", position.as_ref(), error)?;
            }
            Self::UnreadableHeader => {
                f.write_str("unreadable header error")?;
            }
            Self::RecordLength {
                position,
                len,
                expected_len,
            } => {
                fmt_variant_message(f, "unequal record length error", position.as_ref())?;
                f.write_str(" (expected: ")?;
                expected_len.fmt(f)?;
                f.write_str(", found: ")?;
                len.fmt(f)?;
                f.write_str(")")?;
            }
            Self::Deserialize { position, error } => {
                fmt_variant_message_with_error(
                    f,
                    "deserialization error",
                    position.as_ref(),
                    error,
                )?;
            }
            Self::Unknown => {
                f.write_str("unknown error")?;
            }
        }

        Ok(())
    }
}

impl error::Error for ImportCsvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IO(error) => Some(error),
            Self::Utf8 { error, .. } => Some(error),
            Self::UnreadableHeader => None,
            Self::RecordLength { .. } => None,
            Self::Deserialize { error, .. } => Some(error),
            Self::Unknown => None,
        }
    }
}

impl From<csv::Error> for ImportCsvError {
    fn from(err: csv::Error) -> Self {
        use csv::ErrorKind::*;

        match err.into_kind() {
            Io(error) => Self::IO(error),
            Utf8 { pos, err } => Self::Utf8 {
                position: pos,
                error: err,
            },
            UnequalLengths {
                pos,
                expected_len,
                len,
            } => Self::RecordLength {
                position: pos,
                expected_len,
                len,
            },
            Seek => Self::UnreadableHeader,
            Deserialize { pos, err } => Self::Deserialize {
                position: pos,
                error: err,
            },
            Serialize(_) => Self::Unknown,
            _ => Self::Unknown,
        }
    }
}

fn get_csv_header_record<R>(reader: &mut Reader<R>) -> Result<Option<&StringRecord>, csv::Error>
where
    R: io::Read,
{
    let header = reader.headers()?;
    if !header.is_empty() {
        Ok(Some(header))
    } else {
        Ok(None)
    }
}

pub fn import_csv<T, R>(csv: &mut R) -> Result<Vec<T>, ImportCsvError>
where
    R: io::Read,
    T: DeserializeOwned,
{
    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .delimiter(b',')
        .flexible(false)
        .has_headers(true)
        .from_reader(csv);

    let headers = get_csv_header_record(&mut reader)?.cloned();

    let mut rows = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: T = record.deserialize(headers.as_ref())?;
        rows.push(row);
    }

    Ok(rows)
}
