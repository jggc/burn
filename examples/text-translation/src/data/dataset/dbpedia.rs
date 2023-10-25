use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub title: String,   // The title of the item
    pub content: String, // The content of the item
    pub label: usize,    // The label of the item (classification category)
}

pub struct DbPediaDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl DbPediaDataset {
    pub fn map_label(label: usize) -> String {
        match label {
            0 => "Company".to_string(),
            1 => "Educational Institution".to_string(),
            2 => "Artist".to_string(),
            3 => "Athlete".to_string(),
            4 => "Office Holder".to_string(),
            5 => "Mean of Transportation".to_string(),
            6 => "Building".to_string(),
            7 => "Natural Place".to_string(),
            8 => "Village".to_string(),
            9 => "Animal".to_string(),
            10 => "Plant".to_string(),
            11 => "Album".to_string(),
            12 => "Film".to_string(),
            13 => "Written Work".to_string(),
            _ => panic!("Unknown DbPedia class"),
        }
    }
}

impl Dataset<DbPediaItem> for DbPediaDataset {
    fn get(&self, index: usize) -> Option<DbPediaItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaDataset {
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
