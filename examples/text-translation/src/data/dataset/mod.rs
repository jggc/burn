mod dbpedia;
pub use dbpedia::*;
mod dbpedia_pretrain;
pub use dbpedia_pretrain::*;
mod refinedweb;
pub use refinedweb::*;
mod vec;
pub use vec::*;
// mod postgres;
// pub use postgres::*;
mod c4_pretrain;
pub use c4_pretrain::*;
mod dbpedia_guess_title;
pub use dbpedia_guess_title::*;
mod dbpedia_classification;
pub use dbpedia_classification::*;
mod multitask_pretrain;
pub use multitask_pretrain::*;
mod masking;

pub struct TextTranslationDataset;

#[derive(new, Clone, Debug)]
pub struct TextTranslationItem {
    pub input: String,  // The input text
    pub output: String, // The expected output text
}
