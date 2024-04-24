use datafusion::{arrow::error::ArrowError, error::DataFusionError};
use failure::Fail;
use tantivy::TantivyError;
pub type Result<T> = std::result::Result<T, FastErr>;

#[derive(Fail, Debug)]
pub enum FastErr {
    #[fail(display = "IO Err: {}", _0)]
    IoError(#[cause] std::io::Error),
    // #[fail(display = "datafusion Err: {}", error)]
    #[fail(display = "Convert Err: {}", _0)]
    ConvertErr(#[cause] core::convert::Infallible),
    #[fail(display = "Arrow Err: {}", _0)]
    ArrowErr(#[cause] ArrowError),
    #[fail(display = "DataFusionErr: {}", _0)]
    DataFusionErr(#[cause] DataFusionError),
    #[fail(display = "Unimplement: {}", _0)]
    UnimplementErr(String),
    #[fail(display = "SerdeErr: {}", _0)]
    SerdeErr(serde_json::Error),
    #[fail(display = "InternalErr: {}", _0)]
    InternalErr(String),
    #[fail(display = "TantivyErr: {}", _0)]
    TantivyErr(#[cause] TantivyError),
    #[fail(display = "JitErr: {}", _0)]
    JitErr(String),
}

impl From<serde_json::Error> for FastErr {
    fn from(value: serde_json::Error) -> Self {
        FastErr::SerdeErr(value)
    }
}

impl From<std::io::Error> for FastErr {
    fn from(value: std::io::Error) -> Self {
        FastErr::IoError(value)
    }
}

impl From<core::convert::Infallible> for FastErr {
    fn from(err: core::convert::Infallible) -> FastErr {
        FastErr::ConvertErr(err)
    }
}

impl  From<ArrowError> for FastErr {
   fn from(value: ArrowError) -> Self {
       FastErr::ArrowErr(value)
   } 
}

impl From<DataFusionError> for FastErr {
    fn from(value: DataFusionError) -> Self {
        FastErr::DataFusionErr(value)
    }
}

impl From<TantivyError> for FastErr {
    fn from(value: TantivyError) -> Self {
        FastErr::TantivyErr(value)
    }
}