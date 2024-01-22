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

use burn::{
    backend::{libtorch::LibTorchDevice, LibTorch},
    optim::decay::WeightDecayConfig,
};
use text_translation::{data::GuessCountryDataset, training::ExperimentConfig, Gpt2Tokenizer};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::backend::Autodiff<LibTorch<Elem>>;

fn main() {
    let mut config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(512, 1024, 16, 6)
            .with_norm_first(true),
        burn::nn::transformer::TransformerDecoderConfig::new(512, 1024, 16, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-5))),
    );

    config.batch_size = 10;

    let dataset_train = GuessCountryDataset::train();
    let dataset_valid = GuessCountryDataset::valid();

    text_translation::training::train::<Backend, GuessCountryDataset>(
        vec![
            LibTorchDevice::Cuda(0),
            // LibTorchDevice::Cuda(1),
            // LibTorchDevice::Cuda(2),
            // LibTorchDevice::Cuda(3),
        ],
        // vec![
        //   LibTorchDevice::Cpu
        // ],
        dataset_train,
        dataset_valid,
        config,
        Arc::new(Gpt2Tokenizer::default()),
        "/datapool/burn_models/guess-the-country-flant5small-gpttokenizer",
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
