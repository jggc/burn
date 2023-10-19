mod gpt;
mod bert;
pub use gpt::*;
pub use bert::*;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn start_token(&self) -> usize;
    fn end_token(&self) -> usize;
    fn mask_token(&self) -> usize;
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
    fn start_token_value(&self) -> String {
        self.decode(&[self.start_token()])
    }
    fn end_token_value(&self) -> String {
        self.decode(&[self.end_token()])
    }
    fn mask_token_value(&self) -> String {
        self.decode(&[self.end_token()])
    }
}
