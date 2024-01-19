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

use text_translation::Gpt2Tokenizer;

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::autodiff::ADBackendDecorator<burn_tch::TchBackend<Elem>>;

fn main() {
    text_translation::inference::infer::<Backend>(
        burn_tch::TchDevice::Cuda(0),
        // burn_tch::TchDevice::Cuda(1),
        // burn_tch::TchDevice::Cuda(2),
        // burn_tch::TchDevice::Cuda(3),
        // vec![
        //   burn_tch::TchDevice::Cpu
        // ],
        "/tmp/text-translation-train-addressparsing-gpttokenizer",
        Arc::new(Gpt2Tokenizer::default()),
    );
}

// reste 10k warmup
// changer learning rate  -> peak environ 1e-5
// plus gros layers, moins profond : modele moins puissant mais plus facile a entrainer
// batch size -> max memoire
// grad accumulation
// dataset -> augmenter / verifier qualite samples
