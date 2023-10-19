use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::data::TextTranslationItem;
pub type AddressField = Option<String>;

pub struct InputAddress {
    pub id: String,
    pub raw: String,
    pub parsed: String,
}

pub struct CsvDataset<T: DeserializeOwned> {
    iterator: Box<dyn Iterator<Item = T>>,
}

impl<T> CsvDataset<T>
where
    T: DeserializeOwned + 'static,
{
    pub fn new(file_path: &str) -> CsvDataset<T> {
        let reader =
            csv::Reader::from_path(file_path).expect(format!("CSV file {file_path} should exist and be readable").as_str());
        let iterator = reader
            .into_deserialize()
            .map(|item| item.expect("CSV item should deserialize to given type"));

        CsvDataset {
            iterator: Box::new(iterator),
        }
    }
}

impl<T> Iterator for CsvDataset<T>
where
    T: DeserializeOwned,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next().into()
    }
}

#[derive(Serialize, Deserialize)]
pub struct CsvInputAddress {
    hash: String,
    raw: String,
    street: AddressField,
    unit: AddressField,
    city: AddressField,
    state_code: AddressField,
    zip: AddressField,
    country_code: AddressField,
    latitude: AddressField,
    longitude: AddressField,
}

impl From<CsvInputAddress> for TextTranslationItem {
    fn from(value: CsvInputAddress) -> Self {
        TextTranslationItem {
            input: value.raw,
            output: format!(
                "street: {}, unit: {}, municipality: {}, province: {}, postal_code: {}, country: {}",
                content_or_none(value.street),
                content_or_none(value.unit),
                content_or_none(value.city),
                content_or_none(value.state_code),
                content_or_none(value.zip),
                content_or_none(value.country_code),
            ),
        }
    }
}

fn content_or_none(maybe_str: Option<String>) -> String {
    match maybe_str {
        Some(content) => content,
        None => "<none>".into(),
    }
}
