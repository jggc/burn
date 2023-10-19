use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

use super::{masking::mask_words, TextTranslationItem};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct C4Item {
    pub text: String,
    pub timestamp: String,
    pub url: String,
}

pub struct C4DatasetPretrain {
    dataset: SqliteDataset<C4Item>,
}

impl Dataset<TextTranslationItem> for C4DatasetPretrain {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        self.dataset.get(index).map(|item| {
            let (masked, inverse_mask) = mask_words(&item.text);
            TextTranslationItem::new(masked, inverse_mask)
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl C4DatasetPretrain {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("validation")
    }
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<C4Item> = HuggingfaceDatasetLoader::new("c4")
            .with_huggingface_cache_dir("/dataset")
            .with_subset("en")
            .with_base_dir("/datapool/burn_dataset")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}
