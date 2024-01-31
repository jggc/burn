use std::env;
use std::path::Path;

use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::record::FullPrecisionSettings;
use burn::record::NamedMpkFileRecorder;
use burn::record::Recorder;
use burn::tensor::activation::log_softmax;
use burn::tensor::activation::relu;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    norm1: BatchNorm<B, 2>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    norm2: BatchNorm<B, 0>,
    phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/mnist");

        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path, &B::Device::default())
            .expect("Failed to decode state");

        Self::new_with(record)
    }
}

impl<B: Backend> Model<B> {
    pub fn new_with(record: ModelRecord<B>) -> Self {
        let conv1 = Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1);
        let conv2 = Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2);
        let conv3 = Conv2dConfig::new([16, 24], [3, 3]).init_with(record.conv3);
        let norm1 = BatchNormConfig::new(24).init_with(record.norm1);
        let fc1 = LinearConfig::new(11616, 32).init_with(record.fc1);
        let fc2 = LinearConfig::new(32, 10).init_with(record.fc2);
        let norm2 = BatchNormConfig::new(10).init_with(record.norm2);
        Self {
            conv1,
            conv2,
            conv3,
            norm1,
            fc1,
            fc2,
            norm2,
            phantom: core::marker::PhantomData,
        }
    }

    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv1_out1 = self.conv1.forward(input1);
        let relu1_out1 = relu(conv1_out1);
        let conv2_out1 = self.conv2.forward(relu1_out1);
        let relu2_out1 = relu(conv2_out1);
        let conv3_out1 = self.conv3.forward(relu2_out1);
        let relu3_out1 = relu(conv3_out1);
        let norm1_out1 = self.norm1.forward(relu3_out1);
        let flatten1_out1 = norm1_out1.flatten(1, 3);
        let fc1_out1 = self.fc1.forward(flatten1_out1);
        let relu4_out1 = relu(fc1_out1);
        let fc2_out1 = self.fc2.forward(relu4_out1);
        let norm2_out1 = self.norm2.forward(fc2_out1);
        log_softmax(norm2_out1, 1)
    }
}
