use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaskWhere, "../../template/mask/where.wgsl");
kernel_wgsl!(MaskWhereInplace, "../../template/mask/where_inplace.wgsl");

pub fn mask_where<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let num_elems = input.shape.num_elements();
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let kernel = StaticKernel::<
        KernelSettings<MaskWhere, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));
    let mask = WgpuTensor::new(mask.client, mask.device, mask.shape, mask.handle);
    let info = build_info(&[&input, &value, &mask, &output]);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[
            &input.handle,
            &value.handle,
            &mask.handle,
            &output.handle,
            &info_handle,
        ],
    );

    output
}

pub fn mask_where_inplace<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
    reverse: bool,
) -> WgpuTensor<E, D> {
    let kernel = StaticKernel::<
        KernelSettings<MaskWhereInplace, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        input.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));
    let mask = WgpuTensor::new(mask.client, mask.device, mask.shape, mask.handle);
    let mut info = build_info(&[&input, &value, &mask]);
    info.push(match reverse {
        true => 1,
        false => 0,
    });
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &value.handle, &mask.handle, &info_handle],
    );

    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Bool, Distribution, Tensor};

    #[test]
    fn mask_where_should_work_with_multiple_invocations() {
        let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_where::<f32, 3>(
            tensor.into_primitive(),
            mask.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.mask_where(mask_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
    #[test]
    fn mask_where_inplace_direction_1_should_work_with_multiple_invocations() {
        let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_where_inplace::<f32, 3>(
            tensor.into_primitive(),
            mask.into_primitive(),
            value.into_primitive(),
            false,
        ));
        let expected = tensor_ref.mask_where(mask_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[test]
    fn mask_where_inplace_direction_0_should_work_with_multiple_invocation() {
        let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_where_inplace::<f32, 3>(
            value.into_primitive(),
            mask.into_primitive(),
            tensor.into_primitive(),
            true,
        ));
        let expected = tensor_ref.mask_where(mask_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[allow(clippy::type_complexity)]
    fn inputs_mask_where() -> (
        Tensor<TestBackend, 3>,
        Tensor<TestBackend, 3>,
        Tensor<TestBackend, 3, Bool>,
        Tensor<ReferenceBackend, 3>,
        Tensor<ReferenceBackend, 3>,
        Tensor<ReferenceBackend, 3, Bool>,
    ) {
        TestBackend::seed(0);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Default, &device);
        let value = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Default, &device);
        let mask =
            Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0., 1.), &device)
                .lower_equal_elem(0.5);

        let device_ref = Default::default();
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &device_ref);
        let value_ref = Tensor::<ReferenceBackend, 3>::from_data(value.to_data(), &device_ref);
        let mask_ref = Tensor::<ReferenceBackend, 3, Bool>::from_data(mask.to_data(), &device_ref);
        assert_eq!(mask.to_data(), mask_ref.to_data());

        (tensor, value, mask, tensor_ref, value_ref, mask_ref)
    }
}
