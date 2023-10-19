use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

use super::{DbPediaDataset, DbPediaItem, TextTranslationItem};

pub struct DbPediaClassificationDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextTranslationItem> for DbPediaClassificationDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        self.dataset.get(index).map(|item| {
            let input = format!("Title: {} - Content: {}", item.title, item.content);
            TextTranslationItem::new(input, DbPediaDataset::map_label(item.label))
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaClassificationDataset {
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
