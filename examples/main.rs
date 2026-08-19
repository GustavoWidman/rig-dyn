use std::env;

use anyhow::Result;
use rig::{
    OneOrMany,
    client::CompletionClient,
    completion::{CompletionModel as RigCompletionModel, CompletionRequest},
    message::{self, Message},
};
use rig_dyn::Provider;

async fn complete_with_rig_model<M>(model: &M, request: CompletionRequest) -> Result<String>
where
    M: RigCompletionModel,
{
    let response = model.completion(request).await?;
    let content = match response.choice.first() {
        message::AssistantContent::Text(content) => content.text.clone(),
        _ => String::new(),
    };

    Ok(content)
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::OpenAI;
    // get api key from somewhere
    let api_key = env::var("OPENAI_API_KEY").unwrap_or("".to_string());
    let client = provider.client(&api_key, None)?;
    let completion_model = CompletionClient::completion_model(&client, "gpt-4o");

    let request = CompletionRequest {
        model: None,
        additional_params: None,
        chat_history: OneOrMany::one(Message::user("Hello, World!")),
        documents: vec![],
        max_tokens: None,
        output_schema: None,
        preamble: Some("You are a helpful assistant.".to_string()),
        temperature: Some(0.7),
        tool_choice: None,
        tools: vec![],
        record_telemetry_content: false,
    };

    let response = complete_with_rig_model(&completion_model, request).await?;
    println!("{}", response);

    Ok(())
}
