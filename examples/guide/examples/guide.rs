use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use guide::{model::ModelConfig, training::TrainingConfig};

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    guide::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    guide::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MNISTDataset::test()
            .get(42)
            .unwrap(),
    );
}
