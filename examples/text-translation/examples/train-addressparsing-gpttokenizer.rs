// TODO
// create or download dummy dataset
// design dataset format
// code dataset
// code model encoder/decoder
// try it out
// find encoder/decoder sensible settings
// tune
// Brizo data
// OSM data

use std::sync::Arc;

use burn::optim::decay::WeightDecayConfig;
use text_translation::{
    addresses::CleanAddressesDataset, data::{VecDataset, TextTranslationItem}, training::ExperimentConfig,
    Gpt2Tokenizer,
};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::autodiff::ADBackendDecorator<burn_tch::TchBackend<Elem>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(512, 1024, 16, 6)
            .with_norm_first(true),
        burn::nn::transformer::TransformerDecoderConfig::new(512, 1024, 16, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-5))),
    );

    let CleanAddressesDataset {
        train: dataset_train,
        valid: dataset_valid,
    } = CleanAddressesDataset::new("/home/nationtech/devel/clean_addresses.csv");

    text_translation::training::train::<Backend, VecDataset<TextTranslationItem>>(
        vec![
            burn_tch::TchDevice::Cuda(0),
            // burn_tch::TchDevice::Cuda(1),
            // burn_tch::TchDevice::Cuda(2),
            // burn_tch::TchDevice::Cuda(3),
        ],
        // vec![
        //   burn_tch::TchDevice::Cpu
        // ],
        dataset_train,
        dataset_valid,
        config,
        Arc::new(Gpt2Tokenizer::default()),
        "/tmp/text-translation-train-addressparsing-gpttokenizer",
        None,
        // Some(101),
    );
}

// reste 10k warmup
// changer learning rate  -> peak environ 1e-5
// plus gros layers, moins profond : modele moins puissant mais plus facile a entrainer
// batch size -> max memoire
// grad accumulation
// dataset -> augmenter / verifier qualite samples
