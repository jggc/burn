use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{
        self, elemwise_workgroup,
        pool::{build_output_and_info_pool2d, build_pool2d_info},
        KernelSettings, WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaxPool2d, "../../template/pool/max_pool2d.wgsl");
kernel_wgsl!(
    MaxPool2dWithIndicesBackward,
    "../../template/pool/max_pool2d_with_indices_backward.wgsl"
);
kernel_wgsl!(
    MaxPool2dWithIndices,
    "../../template/pool/max_pool2d_with_indices.wgsl"
);

pub(crate) fn max_pool2d<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> WgpuTensor<E, 4> {
    let (info_handle, output) =
        build_output_and_info_pool2d(&x, kernel_size, stride, padding, dilation);
    let kernel = StaticKernel::<
        KernelSettings<MaxPool2d, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client
        .execute(Box::new(kernel), &[&x.handle, &output.handle, &info_handle]);

    output
}

pub(crate) fn max_pool2d_with_indices<E: WgpuElement, I: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (WgpuTensor<E, 4>, WgpuTensor<I, 4>) {
    let (info_handle, output) =
        build_output_and_info_pool2d(&x, kernel_size, stride, padding, dilation);
    let indices = empty_device(x.client.clone(), x.device, output.shape.clone());

    let kernel = StaticKernel::<
        KernelSettings<MaxPool2dWithIndices, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client.execute(
        Box::new(kernel),
        &[&x.handle, &output.handle, &indices.handle, &info_handle],
    );

    (output, indices)
}

pub(crate) fn max_pool2d_with_indices_backward<E: WgpuElement, I: WgpuElement>(
    x: WgpuTensor<E, 4>,
    grad: WgpuTensor<E, 4>,
    indices: WgpuTensor<I, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> WgpuTensor<E, 4> {
    let grad = kernel::into_contiguous(grad);
    let indices = kernel::into_contiguous(indices);

    let num_elems = x.shape.num_elements();
    let buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.client.clone(), x.device.clone(), x.shape.clone(), buffer);

    let info_handle = build_pool2d_info(&x, &grad, kernel_size, stride, padding, dilation);

    let kernel = StaticKernel::<
        KernelSettings<MaxPool2dWithIndicesBackward, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client.execute(
        Box::new(kernel),
        &[&indices.handle, &grad.handle, &output.handle, &info_handle],
    );
    output
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{module, ops::ModuleOps, Distribution, Tensor};

    #[test]
    pub fn max_pool2d_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 4>::random(
            [32, 32, 32, 32],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];
        let dilation = [1, 1];

        let pooled = module::max_pool2d(tensor, kernel_size, stride, padding, dilation);
        let pooled_ref = module::max_pool2d(tensor_ref, kernel_size, stride, padding, dilation);

        pooled
            .into_data()
            .assert_approx_eq(&pooled_ref.into_data(), 3);
    }

    #[test]
    pub fn max_pool2d_with_indices_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 4>::random(
            [32, 32, 32, 32],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];
        let dilation = [1, 1];

        let (pooled, indices) =
            module::max_pool2d_with_indices(tensor, kernel_size, stride, padding, dilation);
        let (pooled_ref, indices_ref) =
            module::max_pool2d_with_indices(tensor_ref, kernel_size, stride, padding, dilation);

        pooled
            .into_data()
            .assert_approx_eq(&pooled_ref.into_data(), 3);
        assert_eq!(indices.into_data(), indices_ref.into_data().convert());
    }

    #[test]
    pub fn max_pool2d_with_indices_backward_should_work_with_multiple_invocations() {
        let test_device = Default::default();
        let tensor =
            Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default, &test_device);
        let grad_output =
            Tensor::<TestBackend, 4>::random([32, 32, 16, 16], Distribution::Default, &test_device);
        let ref_device = Default::default();
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &ref_device);
        let grad_output_ref =
            Tensor::<ReferenceBackend, 4>::from_data(grad_output.to_data(), &ref_device);
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];
        let dilation = [1, 1];

        let (_, indices) =
            module::max_pool2d_with_indices(tensor.clone(), kernel_size, stride, padding, dilation);
        let (_, indices_ref) = module::max_pool2d_with_indices(
            tensor_ref.clone(),
            kernel_size,
            stride,
            padding,
            dilation,
        );
        let grad = TestBackend::max_pool2d_with_indices_backward(
            tensor.into_primitive(),
            kernel_size,
            stride,
            padding,
            dilation,
            grad_output.into_primitive(),
            indices.into_primitive(),
        )
        .x_grad;
        let grad_ref = ReferenceBackend::max_pool2d_with_indices_backward(
            tensor_ref.into_primitive(),
            kernel_size,
            stride,
            padding,
            dilation,
            grad_output_ref.into_primitive(),
            indices_ref.into_primitive(),
        )
        .x_grad;

        Tensor::<TestBackend, 4>::from_primitive(grad)
            .into_data()
            .assert_approx_eq(
                &Tensor::<ReferenceBackend, 4>::from_primitive(grad_ref).into_data(),
                3,
            );
    }
}
