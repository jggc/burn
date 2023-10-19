use burn::data::dataset::Dataset;

use super::{C4DatasetPretrain, DbPediaClassificationDataset, TextTranslationItem, DbPediaWriteTitleDataset};

// TODO :
// Create prefixed task dataset
// prefix
//
// input -> "replace inverse mask : Today <M> beautiful day."
// output -> "<M> is a <M>"
// metric -> Accuracy ?
//
// Load C4 as masked dataset
// Load dbpedia as masked dataset
// Load a bajillion other datasets as masked datasets
//
// Load DBPedia for text classification
// input -> Indicate the DBPedia classification : title...content...
// output -> class
// metric -> Accuracy
//
// input -> Write a title : Last night a man decided to be happy in St-Patente-des-Paul
// output -> Man makes life changing decision in St-Patente-des-Paul
// metric -> Rouge
//
// Load math equations
// input -> Solve math problem : problem....
// output -> solution
// metric -> Accuracy
//
// Translation datasets
// input -> translate english to french : Today is a day
// output -> Aujourd'hui est une journée
// metric -> Bleu
pub struct MultiTaskDataset {
    dataset: Vec<Box<dyn PrefixedTranslationDataset>>,
    len: usize,
    len_per_dataset: usize,
}

pub trait PrefixedTranslationDataset:
    Dataset<TextTranslationItem> + TaskPrefix + Send + Sync
{
    fn get_with_prefix(&self, index: usize) -> Option<TextTranslationItem> {
        let prefix = self.get_prefix();

        match self.get(index) {
            Some(mut item) => {
                item.input = format!("{} : {}", prefix, item.input);
                Some(item)
            }
            None => None,
        }
    }
}

pub trait TaskPrefix {
    /// Returns the task prefix such as
    ///
    /// `Translate english to french`
    ///
    /// Do not include trailing space or column, the formatting will be done later.
    ///
    /// This is meant to use the same kind of training as T5 multitask pretraining, see the T5
    /// paper for more information.
    fn get_prefix(&self) -> String;
}

impl TaskPrefix for DbPediaClassificationDataset {
    fn get_prefix(&self) -> String {
        String::from("Identify DbPedia Classification")
    }
}
impl PrefixedTranslationDataset for DbPediaClassificationDataset {}

impl TaskPrefix for DbPediaWriteTitleDataset {
    fn get_prefix(&self) -> String {
        String::from("Write a title for this content")
    }
}
impl PrefixedTranslationDataset for DbPediaWriteTitleDataset {}

impl TaskPrefix for C4DatasetPretrain {
    fn get_prefix(&self) -> String {
        "Fill the masked word span and output an inverse mask".to_owned()
    }
}
impl PrefixedTranslationDataset for C4DatasetPretrain {}

impl MultiTaskDataset {
    pub fn train() -> MultiTaskDataset {
        let mut dataset: Vec<Box<dyn PrefixedTranslationDataset>> = Vec::new();

        dataset.push(Box::new(DbPediaClassificationDataset::train()));
        dataset.push(Box::new(DbPediaWriteTitleDataset::train()));
        // dataset.push(Box::new(C4DatasetPretrain::train()));

        Self::new(dataset)
    }

    pub fn valid() -> MultiTaskDataset {
        let mut dataset: Vec<Box<dyn PrefixedTranslationDataset>> = Vec::new();

        dataset.push(Box::new(DbPediaClassificationDataset::test()));
        dataset.push(Box::new(DbPediaWriteTitleDataset::test()));
        // dataset.push(Box::new(C4DatasetPretrain::test()));

        Self::new(dataset)
    }

    #[cfg(test)]
    fn dbpedia_variants() -> MultiTaskDataset {
        let mut dataset: Vec<Box<dyn PrefixedTranslationDataset>> = Vec::new();

        dataset.push(Box::new(DbPediaClassificationDataset::train()));
        dataset.push(Box::new(DbPediaWriteTitleDataset::train()));

        Self::new(dataset)
    }

    fn new(dataset: Vec<Box<dyn PrefixedTranslationDataset>>) -> MultiTaskDataset {

        let mut smallest_len: usize = usize::MAX;
        dataset.iter().for_each(|element| {
            let len = element.len();
            if len < smallest_len {
                smallest_len = len;
            }
        });
        let total_len = smallest_len * dataset.len();

        MultiTaskDataset {
            dataset,
            len: total_len,
            len_per_dataset: smallest_len,
        }
    }
}

impl Dataset<TextTranslationItem> for MultiTaskDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        let dataset_index = index / self.len_per_dataset;
        // Use ratio of dataset to get values from the entire length of the dataset. This is
        // important for sorted datasets like DbPedia that are ordrered by label so that we do not
        // always get the same label.
        //
        // Ex : DbPedia dataset has 560k items. If there is a smaller dataset of 10k elements and
        // we do not use a ratio position, we would only train on the first 10k elements of DbPedia
        // which would all be the same class
        let ratio_of_dataset : f32 = (index % self.len_per_dataset) as f32 / self.len_per_dataset as f32;

        let d = self.dataset.get(dataset_index).expect(&format!(
            "Should always find valid dataset index, got {}",
            dataset_index
        ));
        let index_in_dataset = (ratio_of_dataset * d.len() as f32) as usize;
        d.get_with_prefix(index_in_dataset)
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn multitask_dataset_should_resolve_dbpedia() {
        let mtds = MultiTaskDataset::dbpedia_variants();

        println!("{:?}", mtds.get(0).unwrap());
        println!("{:?}", mtds.get(560_000).unwrap());
        println!("{:?}", mtds.get(500_000).unwrap());
        println!("{:?}", mtds.get(1_060_000).unwrap());

        assert!(false);
    }
}
