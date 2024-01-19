use burn::data::dataset::{Dataset, SqliteDataset};

use super::TextTranslationItem;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddressMultiformatItem {
    pub input_str: String,  // The input string
    pub output_str: String, // The output string
}

pub struct AddressMultiformatDataset {
    dataset: SqliteDataset<AddressMultiformatItem>,
}

impl Into<TextTranslationItem> for AddressMultiformatItem {
    fn into(self) -> TextTranslationItem {
        TextTranslationItem::new(self.input_str, self.output_str)
    }
}

impl Dataset<TextTranslationItem> for AddressMultiformatDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        match self.dataset.get(index) {
            Some(item) => Some(item.into()),
            None => None,
        }
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl AddressMultiformatDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn valid() -> Self {
        Self::new("validation")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<AddressMultiformatItem> = SqliteDataset::from_db_file(
            "/datapool/huggingface_llm/data/finetuning_dataset.db",
            split,
        )
        .expect("Dataset file should be available and load to sqlite");
        Self { dataset }
    }
}
