use crate::data::{TextTranslationItem, VecDataset};

use super::{CsvDataset, CsvInputAddress};

pub struct CleanAddressesDataset {
    pub train: VecDataset<TextTranslationItem>,
    pub valid: VecDataset<TextTranslationItem>,
}

impl CleanAddressesDataset {
    pub fn new(clean_addresses_dataset_path: &str) -> CleanAddressesDataset {
        let dataset = CsvDataset::<CsvInputAddress>::new(clean_addresses_dataset_path);
        let mut train: Vec<TextTranslationItem> = Vec::with_capacity(900_000);
        let mut valid: Vec<TextTranslationItem> = Vec::with_capacity(9_000);

        let mut counter = 0;
        for input_address in dataset.skip(1) {
            // skip header line
            if counter % 100 == 0 {
                valid.push(input_address.into());
            } else {
                train.push(input_address.into());
            }

            counter += 1;
        }

        CleanAddressesDataset {
            train: VecDataset::new(train),
            valid: VecDataset::new(valid),
        }
    }

    pub fn new_limit(clean_addresses_dataset_path: &str) -> CleanAddressesDataset {
        let mut dataset = CsvDataset::<CsvInputAddress>::new(clean_addresses_dataset_path);
        let mut train: Vec<TextTranslationItem> = Vec::with_capacity(900_000);
        let mut valid: Vec<TextTranslationItem> = Vec::with_capacity(9_000);

        valid.push(dataset.next().unwrap().into());
        train.push(dataset.next().unwrap().into());

        CleanAddressesDataset {
            train: VecDataset::new(train),
            valid: VecDataset::new(valid),
        }
    }
}
/*

pub struct CleanAddressesBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    max_seq_length: usize,
}

impl<B: Backend> Batcher<InputAddress, TextTranslationBatch<B>> for CleanAddressesBatcher<B> {
    fn batch(&self, items: Vec<InputAddress>) -> TextTranslationBatch<B> {
        let mut input_list = Vec::with_capacity(items.len());
        let mut output_list = Vec::with_capacity(items.len());

        for i in items {
            input_list.push(self.tokenizer.encode(&i.raw, true));
            output_list.push(self.tokenizer.encode(&i.parsed, true));
        }

        let input_mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            input_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        let output_mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            output_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        TextTranslationBatch {
            input: input_mask.tensor.to_device(&self.device),
            input_mask_pad: input_mask.mask.to_device(&self.device),
            output: output_mask.tensor.to_device(&self.device),
            output_mask_pad: output_mask.mask.to_device(&self.device),
        }
    }
}
*/
