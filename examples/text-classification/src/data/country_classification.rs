use burn::data::dataset::{Dataset, SqliteDataset};

use super::TextClassificationItem;
use crate::TextClassificationDataset;

// Stuct for items in Country Classification dataset
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CountryClassificationItem {
    pub input_str: String, // The text for classification
    pub target_class: usize,  // The label of the text (classification category)
}

// Struct for the Country Classification dataset
pub struct CountryClassificationDataset {
    dataset: SqliteDataset<CountryClassificationItem>, // Underlying SQLite dataset
}

// Implement the Dataset trait for the Country Classification dataset
impl Dataset<TextClassificationItem> for CountryClassificationDataset {
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| TextClassificationItem::new(item.input_str, item.target_class))
        //map countryclassificationitem to TextClassificationItem
    }

    // Returns the length of the dataset
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// Implement methods for constructing the Country Classification dataset
impl CountryClassificationDataset {
    /**
    Returns the training portion of the dataset
    */
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Returns the testing portion of the dataset
    pub fn valid() -> Self {
        Self::new("validation")
    }

    /// Constructs the dataset from a split (either "train" or "test")
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<CountryClassificationItem> =
            SqliteDataset::from_db_file("/datapool/huggingface_llm/data/country_classification.db", split)
                .expect("Dataset file should be available and load to sqlite");
        Self { dataset }
    }
}

/// Implements the TextClassificationDataset trait for the Country Classification dataset
impl TextClassificationDataset for CountryClassificationDataset {
    /// Returns the number of unique classes in the dataset
    fn num_classes() -> usize {
        // TODO Use vector length or read from database
        214
    }

    /// Returns the name of a class given its label
    fn class_name(target_class: usize) -> String {
        match target_class {
            0 => "Afghanistan",
            1 => "Albania",
            2 => "Algeria",
            3 => "Andorra",
            4 => "Angola",
            5 => "Antarctica",
            6 => "Antigua and Barbuda",
            7 => "Argentina",
            8 => "Armenia",
            9 => "Aruba",
            10 => "Australia",
            11 => "Austria",
            12 => "Azerbaijan",
            13 => "Bahrain",
            14 => "Bangladesh",
            15 => "Barbados",
            16 => "Belarus",
            17 => "Belgium",
            18 => "Belize",
            19 => "Benin",
            20 => "Bhutan",
            21 => "Bolivia",
            22 => "Bosnia and Herzegovina",
            23 => "Botswana",
            24 => "Brazil",
            25 => "Brunei",
            26 => "Bulgaria",
            27 => "Burkina Faso",
            28 => "Burundi",
            29 => "Cambodia",
            30 => "Cameroon",
            31 => "Canada",
            32 => "Cape Verde",
            33 => "Central African Republic",
            34 => "Chad",
            35 => "Chile",
            36 => "China",
            37 => "China Country",
            38 => "Colombia",
            39 => "Comoros",
            40 => "Costa Rica",
            41 => "Croatia",
            42 => "Cuba",
            43 => "Curacao",
            44 => "Cyprus",
            45 => "Czech Republic",
            46 => "Democratic Republic of the Congo",
            47 => "Denmark",
            48 => "Djibouti",
            49 => "Dominica",
            50 => "Dominican Republic",
            51 => "East Timor",
            52 => "Ecuador",
            53 => "Egypt",
            54 => "El Salvador",
            55 => "Equatorial Guinea",
            56 => "Eritrea",
            57 => "Estonia",
            58 => "Ethiopia",
            59 => "Federated States of Micronesia",
            60 => "Fiji",
            61 => "Finland",
            62 => "France",
            63 => "Gabon",
            64 => "Gambia",
            65 => "Georgia",
            66 => "Germany",
            67 => "Ghana",
            68 => "Greece",
            69 => "Greenland",
            70 => "Grenada",
            71 => "Guatemala",
            72 => "Guernsey",
            73 => "Guinea",
            74 => "Guinea Bissau",
            75 => "Guyana",
            76 => "Haiti",
            77 => "Honduras",
            78 => "Hong Kong S.A.R.",
            79 => "Hong Kong Special Administrative Region",
            80 => "Hungary",
            81 => "Iceland",
            82 => "India",
            83 => "Indonesia",
            84 => "Iran",
            85 => "Iraq",
            86 => "Ireland",
            87 => "Isle of Man",
            88 => "Israel",
            89 => "Italy",
            90 => "Ivory Coast",
            91 => "Jamaica",
            92 => "Japan",
            93 => "Jersey",
            94 => "Jordan",
            95 => "Kazakhstan",
            96 => "Kenya",
            97 => "Kiribati",
            98 => "Kosovo",
            99 => "Kuwait",
            100 => "Kyrgyzstan",
            101 => "Laos",
            102 => "Latvia",
            103 => "Lebanon",
            104 => "Lesotho",
            105 => "Liberia",
            106 => "Libya",
            107 => "Liechtenstein",
            108 => "Lithuania",
            109 => "Luxembourg",
            110 => "Macau S.A.R.",
            111 => "Madagascar",
            112 => "Malawi",
            113 => "Malaysia",
            114 => "Maldives",
            115 => "Mali",
            116 => "Malta",
            117 => "Marshall Islands",
            118 => "Mauritania",
            119 => "Mauritius",
            120 => "Mexico",
            121 => "Moldova",
            122 => "Monaco",
            123 => "Mongolia",
            124 => "Montenegro",
            125 => "Morocco",
            126 => "Mozambique",
            127 => "Myanmar",
            128 => "Namibia",
            129 => "Nauru",
            130 => "Nepal",
            131 => "Netherlands",
            132 => "New Zealand",
            133 => "Nicaragua",
            134 => "Niger",
            135 => "Nigeria",
            136 => "North Korea",
            137 => "North Macedonia",
            138 => "Northern Cyprus",
            139 => "Norway",
            140 => "Oman",
            141 => "Pakistan",
            142 => "Palau",
            143 => "Palestine",
            144 => "Panama",
            145 => "Papua New Guinea",
            146 => "Paraguay",
            147 => "Peru",
            148 => "Philippines",
            149 => "Poland",
            150 => "Portugal",
            151 => "Qatar",
            152 => "Republic of Congo",
            153 => "Romania",
            154 => "Russia",
            155 => "Rwanda",
            156 => "Saint Kitts and Nevis",
            157 => "Saint Lucia",
            158 => "Saint Vincent and the Grenadines",
            159 => "Samoa",
            160 => "San Marino",
            161 => "Sao Tome and Principe",
            162 => "Saudi Arabia",
            163 => "Senegal",
            164 => "Serbia",
            165 => "Seychelles",
            166 => "Sierra Leone",
            167 => "Singapore",
            168 => "Sint Maarten",
            169 => "Slovakia",
            170 => "Slovenia",
            171 => "Solomon Islands",
            172 => "Somalia",
            173 => "Somaliland",
            174 => "South Africa",
            175 => "South Korea",
            176 => "South Sudan",
            177 => "Spain",
            178 => "Spratly Islands",
            179 => "Sri Lanka",
            180 => "Sudan",
            181 => "Suriname",
            182 => "Sweden",
            183 => "Switzerland",
            184 => "Syria",
            185 => "Taiwan",
            186 => "Tajikistan",
            187 => "Tanzania",
            188 => "Thailand",
            189 => "The Bahamas",
            190 => "Togo",
            191 => "Tonga",
            192 => "Trinidad and Tobago",
            193 => "Tunisia",
            194 => "Turkey",
            195 => "Turkmenistan",
            196 => "Tuvalu",
            197 => "Uganda",
            198 => "Ukraine",
            199 => "United Arab Emirates",
            200 => "United Kingdom",
            201 => "United Nations",
            202 => "United States",
            203 => "Uruguay",
            204 => "Uzbekistan",
            205 => "Vanuatu",
            206 => "Vatican",
            207 => "Venezuela",
            208 => "Vietnam",
            209 => "Western Sahara",
            210 => "Yemen",
            211 => "Zambia",
            212 => "Zimbabwe",
            213 => "eSwatini",
            _ => panic!("invalid class"),
        }
        .to_string()
    }
}
