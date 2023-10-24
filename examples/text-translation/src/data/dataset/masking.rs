use rand::thread_rng;
use rand::Rng;
use std::collections::hash_map::Keys;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    static ref DICT: HashMap<String, Vec<String>> = thesaurus::dict();
    static ref DICT_KEYS: Mutex<Keys<'static, String, Vec<String>>> = Mutex::new(DICT.keys());
}

const MASK_TOKEN: &str = " [MASK]";

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
        inverse_mask += MASK_TOKEN;

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
            masked += MASK_TOKEN;
        }
    }
    (masked.trim().into(), inverse_mask)
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn mask_should_modify_string() {
        assert_mask_masks("Hello, today is a beautiful day and I want to go take a long walk with some of my friends, Stacey, Paul and Bertha");
    }

    #[test]
    fn mask_should_work_on_multiline_string() {
        assert_mask_masks(
            r#"Get business insurance in Bolton, now!
Keep your property and your employees protected with business insurance in Bolton, from The Co-operators! The Co-operators can provide you with business insurance in Bolton that takes your specific insurance needs into consideration. Our business insurance in Bolton comes with a level of customer care you won't find anyplace else. That's exactly what you'll receive when you choose The Co-operators to provide your ideal policy. Talk to an Insurance Advisor about a personalized quote today!
Obtaining commercial insurance means considering a wide range of factors and variables. For this reason, it isn’t as easy to compare quotes online as it is with other insurance products. So which insurer can you turn to? When you turn to The Co-operators, you have access to an extended network of expert advisors in Bolton and across the province from a co-operative with more than 70 years of experience assisting Canadian companies just like yours. We're here to guide you through company insurance and help you make the right decisions every step of the way.
A big deal for owners!
As a responsible corporate citizen, we believe in balancing our economic, environmental and social priorities. In fact, The Co-operators was recently recognized in Hewitt Associate's Green 30 guide, which identified Canada's 30 most environmentally-conscious employers. Talk to The Co-operators today about business insurance in Bolton AND about what we're doing to help the environment.
Get extended Business Insurance in Bolton now!
Bolton has a public business policy system that provides a minimum level of mandatory coverage to vehicle owners when they purchase license plates."#,
        );
    }

    #[test]
    fn mask_should_handle_basic_special_characters() {
        assert_mask_masks(
            r#"*If your rack does not have a configuration in it already, skip to step 6.
Note: Basic/Portable mode is a simplified setup mode in CEM+ version 2.0 and above, intended for portable/touring racks. In basic/portable mode for software versions below 3.1.0, the only active DMX is port A input. In software version 3.1.0 and above, DMX B input is also active. 
Once a unit is configured and running in basic/portable mode, anyone can press [√] from the front panel to quickly reset the address for all active DMX inputs.
It will ask for input power. It will be set to 120V in the US. Hit the [√].                                                                                                                                                                                                              
It will prompt to "Set Phase Balance". Choose [Straight-3Phase] for three phase power or [Straight-1Phase] for single phase power.
Note: The number of the rack refers to the number of module slots in the rack NOT circuits. e.g. If you have a portable rack that can hold 24 D20 modules in it for a total of 48 circuits, you have an SP24 rack.
It will prompt to "Set Module Type". Select the correct value and hit the [√]. You can only select one type at this point.
It will ask "Set Dimmer Double". This defaults to [Off]. If you are using Dimmer Doubling, change this to [On]. Then press [√].
It will prompt to "Set DMX Start". Scroll to the correct value and hit the [√].
"#,
        );
    }

    #[test]
    fn mask_should_handle_empty_string() {
        let (masked, inverse_mask) = mask_words("");
        assert_eq!(masked.len(), 0);
        assert_eq!(inverse_mask.len(), 0);
    }

    #[test]
    fn mask_should_handle_arabic_chars() {
        assert_mask_masks(
            r#"قاموس سرياني عربي  ܟܐܒ ܟܐقاموس سرياني عربي  ܟܐܒ ܟܐقاموس سرياني عربي  ܟܐܒ ܟܐقاموس سرياني عربي  ܟܐܒ ܟܐقاموس سرياني عربي  ܟܐܒ ܟܐܒܒܒܒܒقاموس سرياني عربي  ܟܐܒ ܟܐܒ"#,
        );
    }

    #[test]
    fn mask_should_handle_multibyte_chars() {
        assert_mask_masks(
            r#"~ 破� .~ 破 �..~  破�..~  破�..~  破�..~ 破�..~ 破�..~  破�..~ 破�..~ 破�..~ 破�..~ 破�..~ 破�..."#,
        );
    }

    fn assert_mask_masks(to_mask: &str) {
        let (masked, inverse_mask) = mask_words(to_mask);

        println!("\n\n\nmasked: {masked}");
        println!("\n\n\ninverse_mask: {inverse_mask}");

        assert_ne!(to_mask, masked);
        assert_ne!(inverse_mask, masked);
        assert!(masked.contains(MASK_TOKEN));
        assert!(inverse_mask.contains(MASK_TOKEN));
    }
}
