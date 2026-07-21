mod client;
mod provider;
mod traits;

pub use client::Client;
pub use client::RigClientCompletionModelAdapter;
pub use provider::Provider;
pub use traits::{
    CompletionModel, DynEmbeddingModel, RigCompletionModelAdapter, RigEmbeddingModelAdapter,
};
