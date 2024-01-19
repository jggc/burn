use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{
        build_info, elemwise_workgroup, KernelSettings, SourceTemplate, StaticKernelSource,
        WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

kernel_wgsl!(
    RecursiveSumRaw,
    "../../template/reduction/recursive_sum.wgsl"
);
kernel_wgsl!(ReductionDimRaw, "../../template/reduction/reduce_dim.wgsl");
kernel_wgsl!(ReductionArgsRaw, "../../template/reduction/args.wgsl");

pub(crate) struct ArgsMax;
pub(crate) struct ArgsMin;
pub(crate) struct SumDim;
pub(crate) struct MeanDim;

impl StaticKernelSource for SumDim {
    fn source() -> SourceTemplate {
        ReductionDimRaw::source().register("assign", "output[id] = sum;")
    }
}

impl StaticKernelSource for MeanDim {
    fn source() -> SourceTemplate {
        ReductionDimRaw::source()
            .add_template(
                "fn mean_dim(sum: {{ elem }}, dim: u32) -> {{ elem }} { 
    return sum / {{ elem }}(dim);
}",
            )
            .register("assign", "output[id] = mean_dim(sum, shape_dim);")
    }
}

impl StaticKernelSource for ArgsMax {
    fn source() -> SourceTemplate {
        ReductionArgsRaw::source()
            .register("cmp", ">")
            .register("initial", (-32767).to_string())
    }
}

impl StaticKernelSource for ArgsMin {
    fn source() -> SourceTemplate {
        ReductionArgsRaw::source()
            .register("cmp", "<")
            .register("initial", 32767.to_string())
    }
}

/// Sum all elements in the input buffer.
pub fn sum<E: WgpuElement, const D: usize>(input: WgpuTensor<E, D>) -> WgpuTensor<E, 1> {
    let mut input_handle = input.handle;
    let mut workgroup = elemwise_workgroup(input.shape.num_elements(), WORKGROUP_DEFAULT);

    loop {
        let num_invocations = workgroup.num_invocations();
        let handle = input
            .client
            .empty(core::mem::size_of::<E>() * num_invocations);

        let kernel = StaticKernel::<
            KernelSettings<RecursiveSumRaw, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(workgroup);

        input
            .client
            .execute(Box::new(kernel), &[&input_handle, &handle]);

        if num_invocations <= 1 {
            return WgpuTensor::new(input.client, input.device, Shape::new([1]), handle);
        }

        input_handle = handle;
        workgroup = elemwise_workgroup(num_invocations, WORKGROUP_DEFAULT);
    }
}

/// Execute the sum dim kernel.
pub fn sum_dim<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    output: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    reduction_dim::<SumDim, E, D>(input, output, dim)
}

/// Execute the mean dim kernel.
pub fn mean_dim<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    output: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    reduction_dim::<MeanDim, E, D>(input, output, dim)
}

fn reduction_dim<K: StaticKernelSource, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    output: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    let kernel =
        StaticKernel::<KernelSettings<K, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>>::new(
            elemwise_workgroup(output.shape.num_elements(), WORKGROUP_DEFAULT),
        );

    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &output.handle, &info_handle],
    );

    output
}

/// Execute the argmax kernel.
pub fn argmax<E: WgpuElement, I: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<I, D> {
    reduction_args_dim::<ArgsMax, E, I, D>(input, dim)
}

/// Execute the argmin kernel.
pub fn argmin<E: WgpuElement, I: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<I, D> {
    reduction_args_dim::<ArgsMin, E, I, D>(input, dim)
}

fn reduction_args_dim<K: StaticKernelSource, E: WgpuElement, I: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<I, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let num_elems = shape_out.num_elements();
    let buffer = input.client.empty(num_elems * core::mem::size_of::<I>());
    let output = WgpuTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer,
    );

    let kernel =
        StaticKernel::<KernelSettings<K, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>>::new(
            elemwise_workgroup(num_elems, WORKGROUP_DEFAULT),
        );
    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &output.handle, &info_handle],
    );

    WgpuTensor::new(output.client, output.device, output.shape, output.handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kernel::reduce::init_reduce_output,
        tests::{ReferenceBackend, TestBackend},
    };
    use burn_tensor::{Distribution, Int, Tensor};

    #[test]
    fn reduction_sum_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 1>::from_primitive(sum(tensor.into_primitive()));
        let val_ref = tensor_ref.sum();

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_sum_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 2>::from_primitive(reduction_dim::<SumDim, f32, 2>(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.sum_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_args_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 2, Int>::from_primitive(argmax(tensor.into_primitive(), 1));
        let val_ref = tensor_ref.argmax(1);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }
}
