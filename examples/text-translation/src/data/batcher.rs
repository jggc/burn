use std::sync::Arc;

use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Int, Tensor},
};

use crate::Tokenizer;

use super::TextTranslationItem;

#[derive(new)]
pub struct TextTranslationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    max_seq_length: usize,
}

#[derive(Clone, Debug)]
pub struct TextTranslationTrainingBatch<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub input_mask_pad: Tensor<B, 2, Bool>,
    pub decoder_input: Tensor<B, 2, Int>,
    pub decoder_input_mask_pad: Tensor<B, 2, Bool>,
    pub expected: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
pub struct TextTranslationInferenceBatch<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub input_mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<TextTranslationItem, TextTranslationTrainingBatch<B>>
    for TextTranslationBatcher<B>
{
    fn batch(&self, items: Vec<TextTranslationItem>) -> TextTranslationTrainingBatch<B> {
        let mut input_list = Vec::with_capacity(items.len());
        let mut output_list = Vec::with_capacity(items.len());
        let mut expected_list = Vec::with_capacity(items.len());

        for i in items {
            input_list.push(self.tokenizer.encode(&i.input, true));
            let mut output_tokens = self.tokenizer.encode(&i.output, true);
            let mut decoder_input = output_tokens.clone();
            decoder_input.pop(); // Remove last token from the decoder input for teacher forcing
            output_tokens.remove(0); // Remove last token from the expected decoder output for
                                     // teacher forcing
            output_list.push(decoder_input);
            expected_list.push(output_tokens);
        }

        let input_mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            input_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        let decoder_input_mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            output_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        let decoder_output_mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            expected_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        TextTranslationTrainingBatch {
            input: input_mask.tensor.to_device(&self.device),
            input_mask_pad: input_mask.mask.to_device(&self.device),
            decoder_input: decoder_input_mask.tensor.to_device(&self.device),
            decoder_input_mask_pad: decoder_input_mask.mask.to_device(&self.device),
            expected: decoder_output_mask.tensor.to_device(&self.device),
        }
    }
}

impl<B: Backend> Batcher<String, TextTranslationInferenceBatch<B>> for TextTranslationBatcher<B> {
    fn batch(&self, items: Vec<String>) -> TextTranslationInferenceBatch<B> {
        let mut input_list = Vec::with_capacity(items.len());

        for i in items {
            input_list.push(self.tokenizer.encode(&i, true));
        }

        let input_mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            input_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );
        TextTranslationInferenceBatch {
            input: input_mask.tensor.to_device(&self.device),
            input_mask_pad: input_mask.mask.to_device(&self.device),
        }
    }
}
