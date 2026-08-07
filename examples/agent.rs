use std::env;

use anyhow::Result;
use rig::prelude::*;
use rig_dyn::Provider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::OpenAI;
    let api_key = env::var("OPENAI_API_KEY")?;

    let client = provider.client(&api_key, None)?;
    let agent = client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant.")
        .temperature(0.7)
        .build();

    let response = agent.prompt("Hello, World!").await?;
    println!("{response}");

    Ok(())
}
