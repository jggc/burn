use crate::data::tokenizer::Tokenizer;
use burn::{module::Module, train::metric::LearningRateMetric};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use burn::{
    config::Config,
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::SamplerDataset, Dataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::transformer::{TransformerDecoderConfig, TransformerEncoderConfig},
    optim::AdamConfig,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LossMetric},
        LearnerBuilder,
    },
};

use crate::{
    data::{TextTranslationBatcher, TextTranslationItem},
    model::TextTranslationModelConfig,
};

#[derive(Config)]
pub struct ExperimentConfig {
    pub encoder: TransformerEncoderConfig,
    pub decoder: TransformerDecoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 512)]
    pub max_seq_length: usize,
    #[config(default = 16)]
    pub batch_size: usize,
    #[config(default = 60)]
    pub num_epochs: usize,
}

pub fn train<B: ADBackend, D: Dataset<TextTranslationItem> + 'static>(
    devices: Vec<B::Device>,
    dataset_train: D,
    dataset_valid: D,
    config: ExperimentConfig,
    tokenizer: Arc<dyn Tokenizer>,
    artifact_dir: &str,
    // The checkpoint to resume the training from
    checkpoint: Option<usize>,
) {
    let device = &devices[0];
    let tokenizer = tokenizer;
    let batcher_train =
        TextTranslationBatcher::new(tokenizer.clone(), device.clone(), config.max_seq_length);
    let batcher_valid =
        TextTranslationBatcher::new(tokenizer.clone(), device.clone(), config.max_seq_length);

    let model = TextTranslationModelConfig::new(
        config.encoder.clone(),
        config.decoder.clone(),
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        config.max_seq_length,
    )
    .init::<B>();

    println!("Training sample: {:?}", dataset_train.get(10).unwrap());
    println!(
        "Config: max_seq_length {} batch_size {} num_epochs {}",
        config.max_seq_length, config.batch_size, config.num_epochs
    );

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 100_000));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_valid, 1_000));

    let accum = 4;
    let optimizer = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(1.0 / accum as f64)
        .with_warmup_steps(10_000)
        .with_model_size(config.encoder.d_model)
        .init();

    let mut learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices)
        .grads_accumulation(accum)
        .num_epochs(config.num_epochs);

    if let Some(checkpoint) = checkpoint {
        learner = learner.checkpoint(checkpoint);
    }

    let learner = learner.build(model, optimizer, lr_scheduler);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Configuration should be saved without error");

    println!("Starting learner.fit");
    let start = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_valid);
    let duration = Duration::from(Instant::now() - start).as_millis();
    println!("Learner.fit done, took {}", duration);

    DefaultRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
