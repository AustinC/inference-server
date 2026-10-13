use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    routing::get,
};
use candle_core::{IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use rand::RngExt;
use serde::Deserialize;
use std::sync::Arc;

const MAX_NEW_TOKENS: usize = 256;

struct Model {
    llama: candle_transformers::models::llama2_c::Llama,
    config: candle_transformers::models::llama2_c::Config,
    cache_template: candle_transformers::models::llama2_c::Cache,
    tokenizer: tokenizers::Tokenizer,
    eos_token: u32,
    device: candle_core::Device,
}

#[derive(Deserialize)]
struct ChatParams {
    prompt: String,
}

// load model weights into the llama2 transformer architecture implementation from the hugging face
// candle lib, which AFAICT is based on the llama2_c implementation by Andrej Karpathy
fn load_model() -> Result<Model> {
    let client = hf_hub::HFClientSync::new()?;
    let weights_path = client
        .model("karpathy", "tinyllamas")
        .download_file()
        .filename("stories15M.bin")
        .send()?;
    let tokenizer_path = client
        .model("hf-internal-testing", "llama-tokenizer")
        .download_file()
        .filename("tokenizer.json")
        .send()?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    // end of sequence token
    let eos_token = tokenizer
        .token_to_id("</s>")
        .context("tokenizer has no </s> token")?;

    // parse the weights stored in a llama2.c-defined binary format.
    let mut file = std::fs::File::open(weights_path)?;
    let config = candle_transformers::models::llama2_c::Config::from_reader(&mut file)?;
    let device = candle_core::Device::Cpu;
    let weights = candle_transformers::models::llama2_c_weights::TransformerWeights::from_reader(
        &mut file, &config, &device,
    )?;
    let vb = weights.var_builder(&config, &device)?;

    let cache_template =
        candle_transformers::models::llama2_c::Cache::new(true, &config, vb.pp("rot"))?;
    let llama = candle_transformers::models::llama2_c::Llama::load(vb, config.clone())?;
    Ok(Model {
        llama,
        config,
        cache_template,
        tokenizer,
        eos_token,
        device,
    })
}

// inference
fn generate(model: &Model, prompt: &str) -> Result<String> {
    // encode the prompt text into a token id vec.
    let mut tokens = model
        .tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    // cache_template is cloned because it contains an empty KV cache to be filled independently for each request
    let mut cache = model.cache_template.clone();
    let mut sampler = LogitsProcessor::new(rand::rng().random(), Some(0.1), Some(0.9));

    // note that during decode .forward() is not called with the growing sequence each time,
    // instead it just receives a single token (the token sampled from the previous pass),
    // so index_pos is necessary for the model to use the correct positional encoding for that token.
    // forward() additionally receives the KV cache which includes attention data computed by the earlier parts of the sequence.
    let mut index_pos = 0;
    for _ in 0..MAX_NEW_TOKENS {
        // the entire prompt get process on the first aka 'prefill' step, and then just 1 token at a time for 'decode'
        let context_size = if index_pos > 0 { 1 } else { tokens.len() }; // context sixe is 1 for each decode pass
        let start = tokens.len().saturating_sub(context_size); // start index into tokens vec
        let input = Tensor::new(&tokens[start..], &model.device)?;
        // the implementation of forward is built to support batched forward passes on multiple input sequences
        // and it distinguishes the batches along axis 0 of the input tensor.
        // This app just processes one sequence per prequest, no batching, but we still use 'unsqueeze'
        // to add the batch axis as axis 0 of the input tensor to comply with the shape expected by forward().
        let input_batch_of_one = input.unsqueeze(0)?;

        // run model
        let logits = model
            .llama
            .forward(&input_batch_of_one, index_pos, &mut cache)?;

        // get the first-and-only-batch (0) then the logit probabilities for the last position in the sequence
        // this returns a 1D tensor the size of the models vocabulary for this position
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        // increment position by how many tokens we just processed (only 1 during decode)
        // but dim(1) here refers to the size of axis 1 in the input tensor.
        index_pos += input_batch_of_one.dim(1)?;

        let next = sampler.sample(&logits)?;
        if next == model.eos_token {
            break;
        }
        tokens.push(next);
        if tokens.len() >= model.config.seq_len {
            break;
        }
    }

    model
        .tokenizer
        .decode(&tokens, true)
        .map_err(anyhow::Error::msg)
}

async fn chat(
    State(model): State<Arc<Model>>,
    Query(params): Query<ChatParams>,
) -> Result<String, (StatusCode, String)> {
    // Inference is compute-bound, so it runs on a thread pool distinct from the async runtime threads.
    // I assume the model param here is a clone of the Arc reference made for this invocation of the handler.
    // it's ownership is then moved into the generate() inference closure
    tokio::task::spawn_blocking(move || generate(&model, &params.prompt))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

// this tokio macro will create the 'real' non-async main function that in turn runs this
// async main function on the tokio executor
#[tokio::main]
async fn main() -> Result<()> {
    // load the model from from hugging face weights and rust ML libs: tokenizer, candle
    let model = tokio::task::spawn_blocking(load_model)
        .await?
        .context("failed to load model")?;

    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/chat", get(chat))
        // put the model struct into a thread-shareable reference, an 'atomic reference counted pointer'
        // since it will have multiple uncoordinated readers and make it available to each Axum request.
        .with_state(Arc::new(model));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}
