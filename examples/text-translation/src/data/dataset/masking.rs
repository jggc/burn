use rand::thread_rng;
use rand::Rng;
use std::collections::hash_map::Keys;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    static ref DICT: HashMap<String, Vec<String>> = thesaurus::dict();
    static ref DICT_KEYS: Mutex<Keys<'static, String, Vec<String>>> = Mutex::new(DICT.keys());
}

fn get_random_word() -> String {
    let mut keys = DICT_KEYS.lock().unwrap();

    match keys.next() {
        Some(word) => word.clone(),
        None => {
            *keys = DICT.keys();
            drop(keys);
            println!("REEINITIALIZED DICT KEYS");
            get_random_word()
        }
    }
}

pub fn mask_words(output: &str) -> (String, String) {
    let mut split_iter = output.split_whitespace();
    let keeping_dist = rand::distributions::Uniform::new_inclusive(4, 12);
    let masking_dist = rand::distributions::Uniform::new_inclusive(1, 3);
    let mut masking_count = 0;
    let mut masked = String::new();
    let mut inverse_mask = String::new();
    let mut rng = thread_rng();
    'masking: loop {
        let keep_length: usize = rng.sample(keeping_dist);
        let mask_length: usize = rng.sample(masking_dist);

        for _ in 0..keep_length {
            let next = split_iter.next();
            match next {
                Some(word) => masked += &format!(" {}", word),
                None => {
                    break 'masking;
                }
            }
        }
        inverse_mask += "[MASK]";

        masking_count += 1;

        if masking_count % 5 <= 1 && mask_length == 1 {
            masked += &format!(" {}", get_random_word());
            match split_iter.next() {
                Some(word) => inverse_mask += &format!(" {}", word),
                None => break 'masking,
            }
        } else {
            for _ in 0..mask_length {
                match split_iter.next() {
                    Some(word) => inverse_mask += &format!(" {}", word),
                    None => break 'masking,
                }
            }
            masked += " <M>";
        }
    }
    (masked.trim().into(), inverse_mask)
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn mask_should_modify_string() {
        let mut to_mask = String::from("Hello, today is a beautiful day and I want to go take a long walk with some of my friends, Stacey, Paul and Bertha");
        to_mask += &to_mask.clone();
        println!("random word {}", DICT_KEYS.lock().unwrap().next().unwrap());

        let mut keys = DICT_KEYS.lock().unwrap();
        keys.find(|_| false);
        drop(keys);

        let (masked, inverse_mask) = mask_words(&to_mask);
        
        println!("masked: {masked}");
        println!("inverse_mask: {inverse_mask}");

        assert_ne!(to_mask, masked);
        assert_eq!(to_mask, masked);
        assert!(masked.contains("<M>"));
    }
}
