# rig-dyn

A dynamic client-provider abstraction framework for Rust applications on top of `rig-core`.

## Overview

rig-dyn is a Rust library that provides a flexible client-provider architecture for building modular and extensible applications. It leverages asynchronous Rust to deliver a seamless experience when working with various service providers. It enables users to use all of the LLM providers supported by rig-core, without having to repeat similar code for each provider, using a simple and intuitive API. This library is meant to be used with `rig-core`, as it only abstracts the client-provider communication and provides a simpler API for working with LLM providers. It in no way replaces `rig-core` or any of its underlying providers, or performs any kind of optimization or API calls on it's own.

## Features

- 🔄 **Dynamic provider registration and discovery**
- 🔒 **Type-safe client-provider communication**
- ⚡ **Asynchronous API built on top of `async-trait`**
- 🧩 **Extensible architecture through traits**
- 🔌 **Simple plugin system for custom providers**
- 🔧 **Serialization and deserialization with `serde`**

## Installation

Add rig-dyn to your `Cargo.toml`:

```toml
[dependencies]
rig-dyn = "0.1.0"
```

## Usage

### Basic Example

```rust
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
```

### Using the `serde` feature

The `serde` feature enables serialization and deserialization of the `Provider` enum, making it easy to save and load provider configurations from JSON, YAML, or other formats:

```rust
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
```

This feature is particularly useful when building applications that need to store user preferences or when working with configuration files that specify which provider to use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Related Projects

- [rig](https://rig.rs/) - A Rust library for interacting with LLM providers
- [Tokio](https://tokio.rs/) - Asynchronous runtime for Rust
- [async-trait](https://github.com/dtolnay/async-trait) - Asynchronous trait methods in Rust
- [serde](https://serde.rs/) - A powerful and efficient serialization and deserialization framework for Rust

---

Built with ❤️ using Rust
