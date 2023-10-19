use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

use super::TextTranslationItem;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RefinedWebItem {
    pub content: String, // The content of the item
    pub url: String,     // The title of the item
}

pub struct RefinedwebDataset {
    dataset: SqliteDataset<RefinedWebItem>,
}

impl Dataset<TextTranslationItem> for RefinedwebDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        self.dataset.get(index).map(|item| {
            todo!()
            // TextTranslationItem::new(input, output)
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl RefinedwebDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<RefinedWebItem> =
            HuggingfaceDatasetLoader::new("tiiuae/falcon-refinedweb")
                .dataset(split)
                .unwrap();
        Self { dataset }
    }
}
