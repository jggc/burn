use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

use super::{masking::mask_words, TextTranslationItem, DbPediaItem};

pub struct PretrainDbPediaDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextTranslationItem> for PretrainDbPediaDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        self.dataset.get(index).map(|item| {
            let full_content = format!(
                "Title: {} - Content: {} - Label : {}",
                item.title, item.content, item.label
            );
            let (masked, inverse_mask) = mask_words(&full_content);
            TextTranslationItem::new(masked, inverse_mask)
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl PretrainDbPediaDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}
