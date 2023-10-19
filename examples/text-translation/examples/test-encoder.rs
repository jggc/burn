use std::sync::Arc;

use burn::{
    backend::NdArrayBackend,
    tensor::{Shape, Tensor, Int},
};
use text_translation::{BertCasedTokenizer, Tokenizer};
pub type TestBackend = NdArrayBackend<f32>;

fn main() {
    // let t = Tensor::random([2, 2, 2]);
    //

    let tensor = Tensor::<TestBackend, 1, Int>::arange(0..12);
    let tensor = tensor.reshape([1, 3, 4]);
    println!("\nexpecting [[[0,1,2,3],[4,5,6,7],[8,9,10,11]]] : {:?}", tensor); // [[[0,1,2,3],[4,5,6,7],[8,9,10,11]]]
    println!("expecting [1, 3, 1] : {:?}", tensor.dims()); // [1, 3, 1]
                                     //
    let tensor = tensor.reshape([12]);
    println!("\nexpecting [0,1,2,3,4,5,6,7,8,9,10,11] : {:?}", tensor); // [0,1,2,3,4,5,6,7,8,9,10,11]
    println!("expecting [12] : {:?}", tensor.dims()); // [12]
                                     //
    let tensor = tensor.reshape([1, 3, 4]);
    println!("\nexpecting [[[0,1,2,3],[4,5,6,7],[8,9,10,11]]] : {:?}", tensor); // [[[0,1,2,3],[4,5,6,7],[8,9,10,11]]]
    println!("expecting [1, 3, 1] : {:?}", tensor.dims()); // [1, 3, 1]
                                     //
    let tensor_slices = tensor.clone().slice([0..1, 0..3, 1..2]);
    println!("\nexpecting [1, 3, 1] : {:?}", tensor_slices.dims()); // [1, 3, 1]
    println!("expecting [[[1],[5],[9]]] : {:?}", tensor_slices); // [[[1],[5],[9]]]
                                     //
    let tensor_slices = tensor.clone().slice([0..1, 1..3]);
    println!("\nexpecting dim [1, 2, 4] : {:?}", tensor_slices.dims()); // [1, 2, 4]
    println!("expecting [[[4,5,6,7],[8,9,10,11]]] : {:?}", tensor_slices); // [[[4,5,6,7],[8,9,10,11]]]
                                     //
    let tensor_slices = tensor.slice([0..1, 1..3, 1..2]);
    println!("\nexpecting [1, 2, 1] : {:?}", tensor_slices.dims()); // [1, 2, 1]
    println!("expecting [[[5],[9]]] : {:?}", tensor_slices); // [[[5],[9]]]
}
