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
use postgres::{Client, NoTls};
use text_translation::{addresses::VecDataset, training::ExperimentConfig, Gpt2Tokenizer, data::PostgresDataset};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::autodiff::ADBackendDecorator<burn_tch::TchBackend<Elem>>;

fn main() {
    let mut config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::nn::transformer::TransformerDecoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    let mut client = Client::connect(
        format!("postgres://postgres:nationtech@localhost/{}", "burn").as_str(),
        NoTls,
    )
    .expect("Postgres connection should succeed");

    let dataset_train = PostgresDataset::new(client, "refinedweb_masked_train");
    let dataset_valid = PostgresDataset::new(client, "refinedweb_masked_valid");

    text_translation::training::train::<Backend, PostgresDataset>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            burn_tch::TchDevice::Cuda(0)
            // burn_tch::TchDevice::Cpu
        },
        dataset_train,
        dataset_valid,
        config,
        Arc::new(Gpt2Tokenizer::default()),
        "/tmp/text-translation-refinedweb-gpt2tokenizer",
        None,
    );
}
