use burn::{
    tensor::{backend::Backend, Int, Tensor},
    train::metric::{Adaptor, LossInput},
};

pub struct TranslationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2, Int>,
    pub expected: Tensor<B, 2, Int>,
}

// TODO
/*
impl<B: Backend> Adaptor<AccuracyInput<B>> for TranslationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.expected.clone())
    }
}
*/

impl<B: Backend> Adaptor<LossInput<B>> for TranslationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
