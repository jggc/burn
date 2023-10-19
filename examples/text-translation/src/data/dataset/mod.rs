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

fn split_string(input_str: &str, max_length: usize) -> (String, String) {
    let mut mid_index;

    if input_str.len() <= max_length * 2 {
        mid_index = input_str.len() / 2;
    } else {
        mid_index = max_length;
    }

    while !input_str.is_char_boundary(mid_index) {
        mid_index += 1;
    }

    let (first_half, second_half) = input_str.split_at(mid_index);
    (
        first_half.to_string(),
        second_half.chars().take(max_length).collect(),
    )
}
