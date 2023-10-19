use std::sync::Mutex;
use burn::data::dataset::Dataset;
use postgres::{types::Type, Client, Statement};

use super::{masking::mask_words, TextTranslationItem};

pub struct PostgresDataset {
    client: Mutex<Client>,
    get_by_index_statement: Statement,
    get_count_statement: Statement,
}

impl PostgresDataset {
    pub fn new(mut client: Client, dataset_table: &str) -> PostgresDataset {
        let get_by_index_statement = client
            .prepare_typed(
                format!(
                    "SELECT input, output from {} where index = $1",
                    dataset_table
                )
                .as_str(),
                &[Type::INT8],
            )
            .unwrap();
        let get_count_statement = client
            .prepare(format!("SELECT count(*) from {}", dataset_table).as_str())
            .unwrap();
        PostgresDataset {
            client: Mutex::new(client),
            get_by_index_statement,
            get_count_statement,
        }
    }
}

impl Dataset<TextTranslationItem> for PostgresDataset {
    fn get(&self, index: usize) -> Option<TextTranslationItem> {
        let result = self
            .client
            .lock()
            .unwrap()
            .query_opt(&self.get_by_index_statement, &[&(index as i64)])
            .expect("Postgres query should succeed");

        match result {
            Some(row) => {
                let output: String = row.get("output");
                Some(TextTranslationItem {
                    input: mask_words(&output).0,
                    output,
                })
            }
            None => None,
        }
    }

    fn len(&self) -> usize {
        let count: i64 = self
            .client
            .lock()
            .unwrap()
            .query_one(&self.get_count_statement, &[])
            .expect("Postgres query should succeed")
            .get(0);

        count as usize
    }
}

#[cfg(test)]
mod test {

    /*
     * Very flaky tests useful to see if SQL syntax is ok
        const TABLE_NAME: &str = "test_dataset";
        use super::*;

        #[test]
        fn get_returns_row_with_right_index() {
            let client = prepare_db_client();
            let dataset = PostgresDataset::new(client, TABLE_NAME);
            let item: TextTranslationItem = dataset.get(1).unwrap();
            assert_eq!(item.input, "test input");
            assert_eq!(item.output, "test output");
        }

        #[test]
        fn len_returns_proper_count() {
            let client = prepare_db_client();
            let dataset = PostgresDataset::new(client, TABLE_NAME);
            assert_eq!(dataset.len(), 1);
        }

        fn prepare_db_client() -> Client {
            println!("Initializing postgres");
            let mut client = Client::connect(
                format!("postgres://postgres:nationtech@localhost/{}", "burn").as_str(),
                NoTls,
            )
            .expect("Postgres connection should succeed");

            let _ = client.execute(format!("drop table {}", TABLE_NAME).as_str(), &[]);
            let create_result = client.execute(format!("create table {} (index bigserial primary key, input text not null, output text not null)", TABLE_NAME).as_str(), &[]);
            if create_result.is_ok() {
                client
                    .execute(
                        format!(
                        "insert into {TABLE_NAME}(input, output) values ('test input', 'test output')"
                    )
                        .as_str(),
                        &[],
                    )
                    .unwrap();
            }

            client
        }
    */
}
