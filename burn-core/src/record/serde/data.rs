use std::collections::HashMap;

use super::adapter::BurnModuleAdapter;
use super::de::Deserializer;
use super::error::Error;
use super::ser::Serializer;
use crate::record::{PrecisionSettings, Record};
use crate::tensor::backend::Backend;

use regex::Regex;
use serde::Deserialize;

/// The main data structure used for deserialization.
///
/// It can hold tree-like structures of nested maps and vectors.
#[derive(Debug, Clone)]
pub enum NestedValue {
    /// The default value, which actually does not hold any value and it is used to indicate that
    /// the value should be populated with the default value.
    Default,

    /// A boolean value.
    Bool(bool),

    /// A string value.
    String(String),

    /// Floating point 32-bit value.
    F32(f32),

    /// Floating point 64-bit value.
    F64(f64),

    /// Signed 16-bit integer value.
    I16(i16),

    /// Signed 32-bit integer value.
    I32(i32),

    /// Signed 64-bit integer value.
    I64(i64),

    /// Unsigned 16-bit integer value used for bf16 and f16 serialization
    U16(u16),

    /// Unsigned 64-bit integer value.
    U64(u64),

    /// A map of nested values (typically used for structs)
    Map(HashMap<String, NestedValue>),

    /// A vector of nested values (typically used for vector of structs or numbers)
    Vec(Vec<NestedValue>),
}
impl NestedValue {
    /// Get the nested value as a map.
    pub fn as_map(self) -> Option<HashMap<String, NestedValue>> {
        match self {
            NestedValue::Map(map) => Some(map),
            _ => None,
        }
    }

    /// Get the nested value as a boolean.
    pub fn as_bool(self) -> Option<bool> {
        match self {
            NestedValue::Bool(bool) => Some(bool),
            _ => None,
        }
    }

    /// Get the nested value as a string.
    pub fn as_string(self) -> Option<String> {
        match self {
            NestedValue::String(string) => Some(string),
            _ => None,
        }
    }

    /// Get the nested value as a f32.
    pub fn as_f32(self) -> Option<f32> {
        match self {
            NestedValue::F32(f32) => Some(f32),
            _ => None,
        }
    }

    /// Get the nested value as a f64.
    pub fn as_f64(self) -> Option<f64> {
        match self {
            NestedValue::F64(f64) => Some(f64),
            _ => None,
        }
    }

    /// Get the nested value as an i16.
    pub fn as_i16(self) -> Option<i16> {
        match self {
            NestedValue::I16(i16) => Some(i16),
            _ => None,
        }
    }

    /// Get the nested value as an i32.
    pub fn as_i32(self) -> Option<i32> {
        match self {
            NestedValue::I32(i32) => Some(i32),
            _ => None,
        }
    }

    /// Get the nested value as an i64.
    pub fn as_i64(self) -> Option<i64> {
        match self {
            NestedValue::I64(i64) => Some(i64),
            _ => None,
        }
    }

    /// Get the nested value as a u16.
    pub fn as_u16(self) -> Option<u16> {
        match self {
            NestedValue::U16(u16) => Some(u16),
            _ => None,
        }
    }

    /// Get the nested value as a u64.
    pub fn as_u64(self) -> Option<u64> {
        match self {
            NestedValue::U64(u64) => Some(u64),
            _ => None,
        }
    }

    /// Deserialize a nested value into a record type.
    pub fn try_into_record<T, PS, A, B>(self, device: &B::Device) -> Result<T, Error>
    where
        B: Backend,
        T: Record<B>,
        PS: PrecisionSettings,
        A: BurnModuleAdapter,
    {
        let deserializer = Deserializer::<A>::new(self, false);

        let item = T::Item::deserialize(deserializer)?;

        // Convert the deserialized item into a Record instance
        Ok(T::from_item::<PS>(item, device))
    }
}

/// Remap the tensor locations according to the key remapping.
///
/// # Arguments
///
/// * `tensors` - A map of tensors.
/// * `key_remap` - A vector of tuples containing a regular expression and a replacement string.
///                See [regex::Regex::replace](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace)
///                for more information.
///
/// # Returns
///
/// A map of tensors with the remapped keys.
pub fn remap<T>(
    mut tensors: HashMap<String, T>,
    key_remap: Vec<(Regex, String)>,
) -> HashMap<String, T> {
    if key_remap.is_empty() {
        return tensors;
    }

    let mut remapped = HashMap::new();

    for (name, tensor) in tensors.drain() {
        let mut new_name = name.clone();
        for (pattern, replacement) in &key_remap {
            if pattern.is_match(&name) {
                new_name = pattern.replace_all(&name, replacement.as_str()).to_string();
                break;
            }
        }
        remapped.insert(new_name, tensor);
    }

    remapped
}

/// Helper function to insert a value into a nested map/vector of tensors.
fn insert_nested_value(current: &mut NestedValue, keys: &[&str], value: NestedValue) {
    if keys.is_empty() {
        *current = value;
        return;
    }

    match current {
        NestedValue::Map(map) => {
            if !map.contains_key(keys[0]) {
                let next = if keys[1..]
                    .first()
                    .and_then(|k| k.parse::<usize>().ok())
                    .is_some()
                {
                    NestedValue::Vec(Vec::new())
                } else {
                    NestedValue::Map(HashMap::new())
                };
                map.insert(keys[0].to_string(), next);
            }
            insert_nested_value(map.get_mut(keys[0]).unwrap(), &keys[1..], value);
        }
        NestedValue::Vec(vec) => {
            let index = keys[0].parse::<usize>().unwrap();
            if index >= vec.len() {
                vec.resize_with(index + 1, || NestedValue::Map(HashMap::new()));
            }
            insert_nested_value(&mut vec[index], &keys[1..], value);
        }
        _ => panic!("Invalid structure encountered"),
    }
}

/// A trait for encapsulating the serialization logic.
pub trait Serializable {
    /// Serializes the object into a `NestedValue` using the provided `Serializer`.
    /// This method is generic over the precision settings `PS`.
    ///
    /// # Parameters
    /// - `serializer`: The `Serializer` to use for serializing the object.
    ///
    /// # Returns
    /// - `Result<NestedValue, Error>`: The result of serialization.
    ///    Returns a `NestedValue` on success,
    ///    or an `Error` on failure.
    ///
    /// # Type Parameters
    /// - `PS`: The precision settings to use during serialization.
    ///    This is a generic parameter and can be any type
    ///    that implements the `PrecisionSettings` trait.
    fn serialize<PS>(&self, serializer: Serializer) -> Result<NestedValue, Error>
    where
        PS: PrecisionSettings;
}

/// Convert a vector of tensors to a nested value.
pub fn unflatten<PS, T>(input: HashMap<String, T>) -> Result<NestedValue, Error>
where
    PS: PrecisionSettings,
    T: Serializable,
{
    let mut result = NestedValue::Map(HashMap::new());

    for (key, value) in input {
        let parts: Vec<&str> = key.split('.').collect();
        let st = value.serialize::<PS>(Serializer::new())?;

        insert_nested_value(&mut result, &parts, st);
    }

    Ok(result)
}
