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
use text_translation::{training::ExperimentConfig, DbPediaDataset, BertCasedTokenizer};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::autodiff::ADBackendDecorator<burn_tch::TchBackend<Elem>>;

fn main() {
    let tokenizer = Arc::new(BertCasedTokenizer::default());
    text_translation::inference::infer::<Backend>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            burn_tch::TchDevice::Cuda(0)
            // burn_tch::TchDevice::Cpu
        },
        "/tmp/text-translation-db-pedia",
        tokenizer,
    );
}
