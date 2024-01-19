use super::unary;
use crate::{
    codegen::{Item, Operator, Variable},
    element::WgpuElement,
    tensor::WgpuTensor,
    unary,
};

unary!(
    |elem| Operator::Clamp {
        input: Variable::Input(0, Item::Scalar(elem)),
        min_value: Variable::Scalar(0, Item::Scalar(elem)),
        max_value: Variable::Scalar(1, Item::Scalar(elem)),
        out: Variable::Local(0, Item::Scalar(elem)),
    },
    scalar 2
);

pub(crate) fn clamp<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    min_value: E,
    max_value: E,
) -> WgpuTensor<E, D> {
    unary::<Ops<E>, OpsInplace<E>, E, D>(input, Some(&[min_value, max_value]), true)
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn clamp_should_match_reference() {
        let input = Tensor::<TestBackend, 4>::random(
            [1, 5, 32, 32],
            Distribution::Default,
            &Default::default(),
        );
        let input_ref =
            Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &Default::default());

        let output = input.clamp(0.3, 0.7);

        output
            .into_data()
            .assert_approx_eq(&input_ref.clamp(0.3, 0.7).into_data(), 3);
    }
}
