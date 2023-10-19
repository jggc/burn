use burn::data::dataset::Dataset;


#[derive(new)]
pub struct VecDataset<T> {
    data: Vec<T>,
}

impl<T: Send + Sync + Clone> Dataset<T> for VecDataset<T> {
    fn get(&self, index: usize) -> Option<T> {
        match self.data.get(index) {
            Some(item) => Some(item.clone()),
            None => None,
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}
