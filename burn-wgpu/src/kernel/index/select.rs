use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

kernel_wgsl!(IndexSelect, "../../template/index/select.wgsl");
kernel_wgsl!(
    SelectAssignInplace,
    "../../template/index/select_assign_inplace.wgsl"
);

pub(crate) fn select<E: WgpuElement, I: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indices: WgpuTensor<I, 1>,
) -> WgpuTensor<E, D> {
    let mut output_shape = tensor.shape.clone();
    output_shape.dims[dim] = indices.shape.dims[0];

    let num_elems = output_shape.num_elements();
    let output = empty_device(tensor.client.clone(), tensor.device.clone(), output_shape);

    let mut info = build_info(&[&tensor, &output]);
    info.push(dim as u32);

    let info_handle = output.client.create(bytemuck::cast_slice(&info));
    let kernel = StaticKernel::<
        KernelSettings<IndexSelect, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));

    tensor.client.execute(
        Box::new(kernel),
        &[
            &tensor.handle,
            &indices.handle,
            &output.handle,
            &info_handle,
        ],
    );

    output
}

pub(crate) fn select_assign<E: WgpuElement, I: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indices: WgpuTensor<I, 1>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut info = build_info(&[&tensor, &value]);
    let mut strides = [0; D];
    let mut current = 1;
    let mut num_elems_per_workgroup = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
            num_elems_per_workgroup *= tensor.shape.dims[index];
        });

    strides
        .into_iter()
        .for_each(|stride| info.push(stride as u32));

    info.push(dim as u32);

    let info_handle = tensor.client.create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<
        KernelSettings<SelectAssignInplace, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        num_elems_per_workgroup,
        WORKGROUP_DEFAULT,
    ));

    tensor.client.execute(
        Box::new(kernel),
        &[&tensor.handle, &indices.handle, &value.handle, &info_handle],
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn select_should_work_with_multiple_workgroups() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let indices = Tensor::<TestBackend, 1, Int>::arange(0..100, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let indices_ref = Tensor::<ReferenceBackend, 1, Int>::from_data(
            indices.to_data().convert(),
            &Default::default(),
        );

        let actual = select(tensor.into_primitive(), 1, indices.into_primitive());
        let expected = tensor_ref.select(1, indices_ref);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_2d_dim0() {
        select_assign_same_as_ref(0, [256, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_2d_dim1() {
        select_assign_same_as_ref(1, [6, 256]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim0() {
        select_assign_same_as_ref(0, [256, 6, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim1() {
        select_assign_same_as_ref(1, [6, 256, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim2() {
        select_assign_same_as_ref(2, [6, 6, 256]);
    }

    fn select_assign_same_as_ref<const D: usize>(dim: usize, shape: [usize; D]) {
        TestBackend::seed(0);
        let tensor =
            Tensor::<TestBackend, D>::random(shape, Distribution::Default, &Default::default());
        let value =
            Tensor::<TestBackend, D>::random(shape, Distribution::Default, &Default::default());
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape[dim]],
                Distribution::Uniform(0., shape[dim] as f64),
                &Default::default(),
            )
            .into_data()
            .convert(),
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &Default::default());
        let value_ref =
            Tensor::<ReferenceBackend, D>::from_data(value.to_data(), &Default::default());
        let indices_ref = Tensor::<ReferenceBackend, 1, Int>::from_data(
            indices.to_data().convert(),
            &Default::default(),
        );

        let actual = Tensor::<TestBackend, D>::from_primitive(select_assign(
            tensor.into_primitive(),
            dim,
            indices.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.select_assign(dim, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
