use std::env;
use std::sync::Arc;

use axum::extract::State;
use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde::Serialize;
use text_translation::inference::TextTranslationInference;

struct AppState {
    model: TextTranslationInference,
}

#[tokio::main]
async fn main() {
    println!("Hello, world!");

    println!("Building state");
    let model_path = match env::var("BURN_MODEL_PATH") {
        Ok(value) => value,
        Err(_) => "/datapool/burn_models/addressmultiformat-flant5small-gpttokenizer/".to_string(),
    };
    println!("model_path : {}", model_path);
    let state = Arc::new(AppState {
        model: TextTranslationInference::new_cuda_gpt(
            model_path,
        ),
    });

    println!("Tracing subsriber");
    // initialize tracing
    tracing_subscriber::fmt::init();

    println!("Building Router");
    // build our application with a route
    let app = Router::new()
        .route("/", get(root))
        .route("/run_inference", post(parse_address))
        .with_state(state);

    println!("Starting listenet");
    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Serving");
    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with a static string
async fn root(State(state): State<Arc<AppState>>) -> String {
    state.model.infer("some address door 345 street my paul")
}

async fn parse_address(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ParseAddress>,
) -> (StatusCode, Json<ParseAddressResponse>) {
    let output = state.model.infer(&payload.raw);
    // insert your application logic here
    let response = ParseAddressResponse { output };

    // this will be converted into a JSON response
    // with a status code of `201 Created`
    (StatusCode::CREATED, Json(response))
}

// the input to our `create_user` handler
#[derive(Deserialize)]
struct ParseAddress {
    raw: String,
}

// the output to our `create_user` handler
#[derive(Serialize)]
struct ParseAddressResponse {
    output: String,
}
