use crate::data::tokenizer::Tokenizer;
use std::{
    io::{self, Write},
    sync::Arc,
};

use burn::{
    config::Config, data::dataloader::batcher::Batcher, module::Module, record::DefaultRecorder,
    record::Recorder, tensor::backend::Backend,
};

use crate::{
    data::TextTranslationBatcher, model::TextTranslationModelConfig, training::ExperimentConfig,
};

pub fn infer<B: Backend>(device: B::Device, artifact_dir: &str, tokenizer: Arc<dyn Tokenizer>) {
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    let batcher = Arc::new(TextTranslationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    ));

    println!("Loading weights ...");
    let record = DefaultRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Model weigths should load");

    println!("Creating model ...");
    let model = TextTranslationModelConfig::new(
        config.encoder,
        config.decoder,
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        config.max_seq_length,
    )
    .init_with::<B>(record)
    .to_device(&device);

    loop {
        print!("Enter a string (or '\\e' to quit): ");
        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        // Trim whitespace and newline characters from the input
        let input = input.trim();

        if input == "\\e" {
            println!("Exiting the program.");
            break;
        }

        // Call your model inference function here
        let item = batcher.batch(vec![String::from(input)]);
        let result = model.infer(item);
        println!("raw inference result shape : {:?}", result.shape());
        println!("raw inference result : {:?}", result);
        let result = result.reshape([-1]);

        println!("sliced result: {:?}", result);
        let logits = result.into_data();
        println!("logits: value {:?} shape {:?}", logits.value, logits.shape);

        let logits_vec: Vec<u32> = logits.convert::<u32>().value;
        let logits_vec: Vec<usize> = logits_vec.into_iter().map(|x: u32| x as usize).collect();
        let decoded = tokenizer.decode(&logits_vec);
        // Print the inference result
        // Print the inference result
        println!("Inference result: value {decoded}");
    }
}
