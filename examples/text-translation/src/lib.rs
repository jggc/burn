#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate lazy_static;

pub mod data;
mod model;
mod learner;
pub mod addresses;

// pub mod inference;
pub mod training;
pub mod inference;

pub use data::TextTranslationDataset;
pub use data::DbPediaDataset;
pub use data::PretrainDbPediaDataset;
pub use data::tokenizer::*;
