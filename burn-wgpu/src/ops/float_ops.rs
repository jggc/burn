use super::numeric;
use crate::codegen::{Elem, Item, Operator, Variable};
#[cfg(not(feature = "autotune"))]
use crate::kernel::matmul::init_matmul_output;
#[cfg(feature = "autotune")]
use crate::kernel::matmul::matmul_autotune;
#[cfg(not(feature = "autotune"))]
use crate::kernel::matmul::vec4::matmul_tiling_2d_vec4;
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
#[cfg(not(feature = "autotune"))]
use crate::kernel::reduce::init_reduce_output;
use crate::kernel::{self, reduce};
use crate::WgpuDevice;
use crate::{unary, FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_tensor::ops::{
    BoolTensor, Device, FloatElem, FloatTensor, FullPrecisionBackend, IntTensor,
};
use burn_tensor::{ops::FloatTensorOps, Data, Distribution, Shape};
use burn_tensor::{ElementConversion, Reader};
use std::ops::Range;

impl<G, F, I> FloatTensorOps<Wgpu<G, F, I>> for Wgpu<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn float_from_data<const D: usize>(
        data: Data<FloatElem<Self>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        super::from_data::<G, F, D>(data, device)
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        match distribution {
            Distribution::Default => random_uniform::<G, F, D>(shape, device, 0.elem(), 1.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform::<G, F, D>(shape, device, low.elem(), high.elem())
            }
            Distribution::Bernoulli(prob) => {
                random_bernoulli::<G, F, D>(shape, device, prob.elem())
            }
            Distribution::Normal(mean, std) => {
                random_normal::<G, F, D>(shape, device, mean.elem(), std.elem())
            }
        }
    }

    fn float_shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn float_into_data<const D: usize>(
        tensor: FloatTensor<Self, D>,
    ) -> Reader<Data<FloatElem<Self>, D>> {
        super::into_data(tensor)
    }

    fn float_device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        super::to_device::<G, F, D>(tensor, device)
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        super::empty::<G, F, D>(shape, device)
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::add(lhs, rhs)
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::add_scalar(lhs, rhs)
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        numeric::zeros::<G, F, D>(shape, device)
    }

    fn float_full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &WgpuDevice,
    ) -> FloatTensor<Self, D> {
        numeric::full::<G, F, D>(shape, device, fill_value)
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        numeric::ones::<G, F, D>(shape, device)
    }

    fn float_sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::sub(lhs, rhs)
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::sub_scalar(lhs, rhs)
    }

    fn float_mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::mul(lhs, rhs)
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::mul_scalar(lhs, rhs)
    }

    fn float_div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::div(lhs, rhs)
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::div_scalar(lhs, rhs)
    }

    fn float_matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[cfg(feature = "autotune")]
        {
            matmul_autotune(lhs, rhs)
        }

        #[cfg(not(feature = "autotune"))]
        {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_vec4(lhs, rhs, out)
        }
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        super::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::gather(dim, tensor, indices)
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::scatter(dim, tensor, indices, value)
    }

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        kernel::select(tensor, dim, indices)
    }

    fn float_select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::select_assign(tensor, dim, indices, value)
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::mask_where(tensor, mask, value)
    }

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        kernel::mask_fill(tensor, mask, value)
    }

    fn float_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal(lhs, rhs)
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::equal_elem(lhs, rhs)
    }

    fn float_greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater(lhs, rhs)
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_elem(lhs, rhs)
    }

    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal_elem(lhs, rhs)
    }

    fn float_lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower(lhs, rhs)
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_elem(lhs, rhs)
    }

    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal_elem(lhs, rhs)
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        reduce::sum(tensor)
    }

    fn float_sum_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        #[cfg(feature = "autotune")]
        {
            reduce::sum_dim_autotune(tensor, dim)
        }

        #[cfg(not(feature = "autotune"))]
        {
            let output = init_reduce_output(&tensor, dim);
            reduce::sum_dim(tensor, output, dim)
        }
    }

    fn float_mean_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        #[cfg(feature = "autotune")]
        {
            reduce::mean_dim_autotune(tensor, dim)
        }

        #[cfg(not(feature = "autotune"))]
        {
            let output = init_reduce_output(&tensor, dim);
            reduce::mean_dim(tensor, output, dim)
        }
    }

    fn float_to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        kernel::cast(tensor.clone())
    }

    fn float_from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        kernel::cast(tensor)
    }

    fn float_exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Exp {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Log {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Log1p {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_powf_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: f32,
    ) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Powf {
                lhs: Variable::Input(0, Item::Scalar(elem)),
                rhs: Variable::Scalar(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: lhs; rhs.elem(),
            elem: F
        )
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Sqrt {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Abs {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Cos {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Sin {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Tanh {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Erf {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_cat<const D: usize>(
        tensors: Vec<FloatTensor<Self, D>>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        kernel::cat(tensors, dim)
    }

    fn float_argmax<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        reduce::argmax(tensor, dim)
    }

    fn float_argmin<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        reduce::argmin(tensor, dim)
    }

    fn float_into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        kernel::cast(tensor)
    }

    fn float_clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        kernel::clamp(tensor, min, max)
    }

    fn float_recip<const D: usize>(
        tensor: FloatTensor<Wgpu<G, F, I>, D>,
    ) -> FloatTensor<Wgpu<G, F, I>, D> {
        unary!(
            operator: |elem: Elem| Operator::Recip {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            },
            input: tensor,
            elem: F
        )
    }

    fn float_repeat<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<Self, D> {
        kernel::repeat(tensor, dim, times)
    }

    fn float_powf<const D: usize>(
        lhs: FloatTensor<Wgpu<G, F, I>, D>,
        rhs: FloatTensor<Wgpu<G, F, I>, D>,
    ) -> FloatTensor<Wgpu<G, F, I>, D> {
        numeric::pow(lhs, rhs)
    }
}
