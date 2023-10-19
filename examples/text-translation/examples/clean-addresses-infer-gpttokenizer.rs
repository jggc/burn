use std::{sync::Arc, env};

use text_translation::Gpt2Tokenizer;

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::autodiff::ADBackendDecorator<burn_tch::TchBackend<Elem>>;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_folder_path = &args[1];
    text_translation::inference::infer::<Backend>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            // burn_tch::TchDevice::Cuda(0)
            burn_tch::TchDevice::Cpu
        },
        // "/tmp/text-translation-clean-addresses-gpt2tokenizer",
        // "/home/jeangab/work/github/burn_models/text-translation-c4-gpt2tokenizer",
        model_folder_path,
        Arc::new(Gpt2Tokenizer::default()),
    );
}
