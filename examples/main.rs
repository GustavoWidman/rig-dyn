use std::env;

use anyhow::Result;
use rig::{
    completion::CompletionRequest,
    message::{self, Message},
};
use rig_dyn::Provider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::OpenAI;
    // get api key from somewhere
    let api_key = env::var("OPENAI_API_KEY").unwrap();
    let client = provider.client(&api_key, None)?;
    let completion_model = client.completion_model("gpt-4o").await;

    let request = CompletionRequest {
        additional_params: None,
        chat_history: vec![],
        documents: vec![],
        max_tokens: None,
        preamble: Some("You are a helpful assistant.".to_string()),
        temperature: Some(0.7),
        tools: vec![],
        prompt: Message::user("Hello, World!"),
    };

    let response = completion_model.completion(request).await?.first();

    match response {
        message::AssistantContent::Text(content) => {
            println!("{}", content.text);
        }
        _ => {}
    }

    Ok(())
}
