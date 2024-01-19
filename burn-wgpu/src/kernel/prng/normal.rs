use burn_tensor::Shape;

use crate::{
    compute::{compute_client, StaticKernel},
    element::WgpuElement,
    kernel::{
        prng::base::{make_args_buffer, make_info_buffer},
        prng_workgroup, KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};

use super::base::Prng;

struct NormalPrng;

impl StaticKernelSource for NormalPrng {
    fn source() -> SourceTemplate {
        Prng::source()
            .register("num_args", "2")
            .register(
                "prng_loop",
                include_str!("../../template/prng/normal_inner_loop.wgsl"),
            )
            .add_template(include_str!(
                "../../template/prng/box_muller_transform.wgsl"
            ))
    }
}

/// Pseudo-random generator for normal distribution
pub fn random_normal<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    mean: E,
    std: E,
) -> WgpuTensor<E, D> {
    const N_VALUES_PER_THREAD: usize = 128; // must be even

    let client = compute_client::<G>(device);
    let output = empty_device(client.clone(), device.clone(), shape.clone());
    let info_handle = make_info_buffer(client.clone(), N_VALUES_PER_THREAD);
    let args_handle = make_args_buffer(client.clone(), &[mean, std]);
    let workgroup = prng_workgroup(shape.num_elements(), WORKGROUP_DEFAULT, N_VALUES_PER_THREAD);
    let kernel = StaticKernel::<
        KernelSettings<NormalPrng, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(workgroup);

    client.execute(
        Box::new(kernel),
        &[&output.handle, &info_handle, &args_handle],
    );

    output
}

#[cfg(test)]
mod tests {

    use burn_tensor::{backend::Backend, Data, Distribution, Shape, Tensor};
    use serial_test::serial;

    use crate::{kernel::prng::base::tests::calculate_bin_stats, tests::TestBackend, WgpuDevice};

    #[test]
    #[serial]
    fn subsequent_calls_give_different_tensors() {
        TestBackend::seed(0);
        let shape = [4, 5];
        let device = WgpuDevice::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Normal(0., 1.), &device);
        let tensor_2 =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Normal(0., 1.), &device);
        for i in 0..20 {
            assert!(tensor_1.to_data().value[i] != tensor_2.to_data().value[i]);
        }
    }

    #[test]
    #[serial]
    fn empirical_mean_close_to_expectation() {
        TestBackend::seed(0);
        let shape = [128, 128];
        let device = WgpuDevice::default();
        let mean = 10.;
        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Normal(mean, 2.), &device);
        let empirical_mean = tensor.mean().into_data();
        empirical_mean.assert_approx_eq(&Data::from([mean as f32]), 1);
    }

    #[test]
    #[serial]
    fn normal_respects_68_95_99_rule() {
        // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        let shape: Shape<2> = [1000, 1000].into();
        let device = WgpuDevice::default();
        let mu = 0.;
        let s = 1.;
        let tensor =
            Tensor::<TestBackend, 2>::random(shape.clone(), Distribution::Normal(mu, s), &device);
        let stats = calculate_bin_stats(
            tensor.into_data().value,
            6,
            (mu - 3. * s) as f32,
            (mu + 3. * s) as f32,
        );
        let assert_approx_eq = |count, percent| {
            let expected = percent * shape.num_elements() as f32 / 100.;
            assert!(f32::abs(count as f32 - expected) < 2000.);
        };
        assert_approx_eq(stats[0].count, 2.1);
        assert_approx_eq(stats[1].count, 13.6);
        assert_approx_eq(stats[2].count, 34.1);
        assert_approx_eq(stats[3].count, 34.1);
        assert_approx_eq(stats[4].count, 13.6);
        assert_approx_eq(stats[5].count, 2.1);
    }
}
