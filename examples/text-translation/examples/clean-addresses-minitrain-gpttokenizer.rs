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
    addresses::CleanAddressesDataset, data::{TextTranslationItem, VecDataset}, training::ExperimentConfig,
    Gpt2Tokenizer,
};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::autodiff::ADBackendDecorator<burn_tch::TchBackend<Elem>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::nn::transformer::TransformerDecoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    let CleanAddressesDataset { train, valid } =
        CleanAddressesDataset::new_limit("/home/jeangab/clean-addresses.csv");
    println!("loaded dataset");

    text_translation::training::train::<Backend, VecDataset<TextTranslationItem>>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            // burn_tch::TchDevice::Cuda(0)
            burn_tch::TchDevice::Cpu
        },
        train,
        valid,
        config,
        Arc::new(Gpt2Tokenizer::default()),
        "/tmp/text-translation-clean-addresses-minitrain-gpt2tokenizer",
        None,
    );
}
