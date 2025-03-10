// Enable the serde feature in your Cargo.toml
// [dependencies]
// rig-dyn = { version = "0.1.0", features = ["serde"] }

use anyhow::Result;
use rig_dyn::Provider;
use serde_plain::{from_str, to_string};

fn main() -> Result<()> {
    // Serialize a provider to a string
    let provider = Provider::OpenAI;
    let serialized = to_string(&provider)?;
    println!("Serialized: {}", serialized); // Outputs: "openai"

    // Deserialize from a string
    let deserialized: Provider = from_str("openai")?;
    assert_eq!(deserialized, Provider::OpenAI);

    // The Provider enum supports various aliases for compatibility
    let from_alias: Provider = from_str("openai-compatible")?;
    assert_eq!(from_alias, Provider::OpenAI);

    // Convert from String using TryFrom
    let from_string = Provider::try_from("anthropic".to_string())?;
    assert_eq!(from_string, Provider::Anthropic);

    Ok(())
}
