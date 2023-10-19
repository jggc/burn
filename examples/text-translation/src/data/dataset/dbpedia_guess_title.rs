use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

use super::{DbPediaItem, DbPediaDataset, TextTranslationItem};

pub struct DbPediaWriteTitleDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextTranslationItem> for DbPediaWriteTitleDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        self.dataset.get(index).map(|item| {
            let input = format!(
                "{} : {}",
                DbPediaDataset::map_label(item.label),
                item.content,
            );
            TextTranslationItem::new(input, item.title)
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaWriteTitleDataset {
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
