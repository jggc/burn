use std::time::{Duration, Instant};

use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLoss,
        transformer::{
            TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput,
            TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
        },
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::{TextTranslationInferenceBatch, TextTranslationTrainingBatch};

#[derive(Config)]
pub struct TextTranslationModelConfig {
    encoder: TransformerEncoderConfig,
    decoder: TransformerDecoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct TextTranslationModel<B: Backend> {
    encoder: TransformerEncoder<B>,
    decoder: TransformerDecoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    // decoder_start_of_sequence: Tensor<B, 3>,
    pad_token: usize,
    vocab_size: usize,
    max_seq_length: usize,
    decoder_d_model: usize,
}

impl TextTranslationModelConfig {
    /// Initialize model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextTranslationModel<B> {
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.encoder.d_model).init(device);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.encoder.d_model).init(device);
        let encoder = self.encoder.init(device);
        let decoder = self.decoder.init(device);
        // TODO not sure about self.vocab_size in the output layer
        let output = LinearConfig::new(self.decoder.d_model, self.vocab_size).init(device);

        TextTranslationModel {
            vocab_size: self.vocab_size,
            encoder,
            decoder,
            embedding_token,
            embedding_pos,
            output,
            decoder_d_model: self.decoder.d_model,
            pad_token: self.pad_token,
            // decoder_start_of_sequence,
            max_seq_length: self.max_seq_length,
        }
    }

    pub(crate) fn init_with<B: Backend>(
        &self,
        record: TextTranslationModelRecord<B>,
    ) -> TextTranslationModel<B> {
        let output =
            LinearConfig::new(self.encoder.d_model, self.vocab_size).init_with(record.output);
        let encoder = self.encoder.init_with(record.encoder);
        let decoder = self.decoder.init_with(record.decoder);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.encoder.d_model)
            .init_with(record.embedding_token);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.encoder.d_model)
            .init_with(record.embedding_pos);

        TextTranslationModel {
            encoder,
            decoder,
            embedding_token,
            embedding_pos,
            output,
            pad_token: self.pad_token,
            vocab_size: self.vocab_size,
            max_seq_length: self.max_seq_length,
            decoder_d_model: self.decoder.d_model,
        }
    }
}

impl<B: Backend> TextTranslationModel<B> {
    pub fn forward_train(&self, item: TextTranslationTrainingBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.input.dims();
        let device = &self.embedding_token.devices()[0];

        let input = item.input;
        let input_mask_pad = item.input_mask_pad;
        let decoder_input = item.decoder_input;

        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length]) // [[0..511]]
            .repeat(0, batch_size); // [[0..511]..batch_size times]

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(input);
        let embedding = (embedding_positions + embedding_tokens) / 2; // This creates position
                                                                      // aware token embeddings
                                                                      // that can be fed directly
                                                                      // into the encoder

        let memory = self
            .encoder
            .forward(TransformerEncoderInput::new(embedding).mask_pad(input_mask_pad.clone()));

        let [batch_size, seq_length] = decoder_input.dims();
        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(decoder_input.clone());
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let input = TransformerDecoderInput::new(embedding, memory)
            .target_mask_attn(mask_attn)
            .target_mask_pad(item.decoder_input_mask_pad)
            .memory_mask_pad(input_mask_pad);

        let decoded = self.decoder.forward(input);

        let output = self.output.forward(decoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        let expected_flatten = item.expected.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLoss::new(Some(self.pad_token), device);
        let loss = loss.forward(output_flatten.clone(), expected_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: expected_flatten,
        }
    }

    pub fn infer(&self, item: TextTranslationInferenceBatch<B>) -> Tensor<B, 2, Int> {
        let [batch_size, seq_length] = item.input.dims();
        let device = &self.embedding_token.devices()[0];

        let input = item.input;
        let input_mask_pad = item.input_mask_pad;

        let index_positions = Tensor::arange(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(input);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let memory = self
            .encoder
            .forward(TransformerEncoderInput::new(embedding).mask_pad(input_mask_pad.clone()));

        let start_token = 50257;
        let mut predictions: Tensor<B, 2, Int> = Tensor::from_ints([[start_token]], device)
            .repeat(0, batch_size)
            .to_device(device); // [batch_size, 1]
        let mut cache = self.decoder.new_autoregressive_cache();
        // for _i in 1..self.max_seq_length + 1 {
        let start = Instant::now();
        for _i in 1..60 + 1 {
            let [batch_size, seq_length] = predictions.dims();
            let index_positions = Tensor::arange(0..seq_length, device)
                .reshape([1, seq_length])
                .repeat(0, batch_size);

            let embedding_positions = self.embedding_pos.forward(index_positions);
            let embedding_tokens = self.embedding_token.forward(predictions.clone());
            let embedding = (embedding_positions + embedding_tokens) / 2;
            let mask_attn = generate_autoregressive_mask(batch_size, seq_length, device);
            let input =
                TransformerDecoderInput::new(embedding, memory.clone()).target_mask_attn(mask_attn);

            let decoded = self
                .decoder
                .forward_autoregressive_inference(input, &mut cache);

            let output = self.output.forward(decoded); // output shape
                                                       // [batch_size, seq_length, vocab_size]
            let next_predictions = output.argmax(2); // [batch_size, seq_length, 1]
            predictions = Tensor::cat(
                vec![
                    predictions,
                    next_predictions
                        .slice([0..batch_size, seq_length - 1..seq_length])
                        .reshape([batch_size, 1]),
                ],
                1,
            );
        }
        let duration = Duration::as_millis(&(Instant::now() - start));
        println!("Inference took {duration} ");
        predictions
        /*
        let [batch_size, seq_length] = expected.dims();
        // let seq_length = seq_length + 10; Nope, repeating token [START]
        println!("expected dims batch_size {batch_size} seq length {seq_length}");
        let index_positions = Tensor::arange_device(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(expected.clone());
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);

        let input = TransformerDecoderInput::new(embedding, memory);
        //.target_mask_attn(mask_attn)
        //.target_mask_pad(item.output_mask_pad)
        //.memory_mask_pad(input_mask_pad);

        let mut cache = self.decoder.new_autoregressive_cache();
        let decoded = self
            .decoder
            .forward_autoregressive_inference(input, &mut cache);

        let output = self.output.forward(decoded);
        println!("shape after output layer {:?}", output.shape());
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        println!("shape output_flatten {:?}", output_flatten.shape());

        let probabilities = softmax(output_flatten, 1);
        println!("shape probabilities {:?}", probabilities.shape());
        probabilities.argmax(1)
        */
    }
}

impl<B: AutodiffBackend> TrainStep<TextTranslationTrainingBatch<B>, ClassificationOutput<B>>
    for TextTranslationModel<B>
{
    fn step(
        &self,
        item: TextTranslationTrainingBatch<B>,
    ) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_train(item);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TextTranslationTrainingBatch<B>, ClassificationOutput<B>>
    for TextTranslationModel<B>
{
    fn step(&self, item: TextTranslationTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward_train(item)
    }
}
