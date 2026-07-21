use async_trait::async_trait;
use rig::{
    completion::{
        self, CompletionError, CompletionRequest, CompletionResponse, GetTokenUsage, Usage,
    },
    embeddings::{self, Embedding, EmbeddingError},
    streaming::StreamingCompletionResponse,
};

use serde::{Deserialize, Serialize};

use embeddings::EmbeddingModel;
use rig::wasm_compat::WasmCompatSend;
use std::sync::Arc;

/// Provides compatibility with previous rig-dyn version (v1.0.3)
/// Note: FinalCompletionResponse was part of rig crate, but it was removed after v0.35.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for FinalCompletionResponse {
    fn token_usage(&self) -> Usage {
        self.usage
    }
}

#[async_trait]
pub trait DynEmbeddingModel: Send + Sync {
    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError>;
    async fn embed_texts(&self, input: Vec<String>) -> Result<Vec<Embedding>, EmbeddingError>;
    fn ndims(&self) -> usize;
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct RigEmbeddingModelAdapter {
    inner: Arc<dyn DynEmbeddingModel>,
}

impl RigEmbeddingModelAdapter {
    #[allow(dead_code)]
    pub fn new(inner: Arc<dyn DynEmbeddingModel>) -> Self {
        Self { inner }
    }
}

impl From<Box<dyn DynEmbeddingModel>> for RigEmbeddingModelAdapter {
    fn from(value: Box<dyn DynEmbeddingModel>) -> Self {
        Self {
            inner: Arc::from(value),
        }
    }
}

impl From<Arc<dyn DynEmbeddingModel>> for RigEmbeddingModelAdapter {
    fn from(value: Arc<dyn DynEmbeddingModel>) -> Self {
        Self { inner: value }
    }
}

impl EmbeddingModel for RigEmbeddingModelAdapter {
    const MAX_DOCUMENTS: usize = 1000;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        panic!("make() is not supported by rig_dyn::EmbeddingModel adapter");
    }

    fn ndims(&self) -> usize {
        self.inner.ndims()
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let texts_vec: Vec<String> = texts.into_iter().collect();
        self.inner.embed_texts(texts_vec).await
    }

    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError> {
        self.inner.embed_text(input).await
    }
}

#[async_trait]
impl<T> DynEmbeddingModel for T
where
    T: EmbeddingModel + Send + Sync,
{
    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError> {
        EmbeddingModel::embed_text(self, input).await
    }

    async fn embed_texts(&self, input: Vec<String>) -> Result<Vec<Embedding>, EmbeddingError> {
        EmbeddingModel::embed_texts(self, input).await
    }

    fn ndims(&self) -> usize {
        EmbeddingModel::ndims(self)
    }
}

#[async_trait]
pub trait CompletionModel: Send + Sync {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError>;
}

#[derive(Clone)]
pub struct RigCompletionModelAdapter {
    inner: Arc<dyn CompletionModel>,
}

impl RigCompletionModelAdapter {
    pub fn new(inner: Arc<dyn CompletionModel>) -> Self {
        Self { inner }
    }
}

impl From<Box<dyn CompletionModel>> for RigCompletionModelAdapter {
    fn from(value: Box<dyn CompletionModel>) -> Self {
        Self {
            inner: Arc::from(value),
        }
    }
}

impl From<Arc<dyn CompletionModel>> for RigCompletionModelAdapter {
    fn from(value: Arc<dyn CompletionModel>) -> Self {
        Self { inner: value }
    }
}

impl completion::CompletionModel for RigCompletionModelAdapter {
    type Response = ();
    type StreamingResponse = FinalCompletionResponse;
    type Client = Arc<dyn CompletionModel>;

    fn make(client: &Self::Client, _model: impl Into<String>) -> Self {
        Self {
            inner: client.clone(),
        }
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + rig::wasm_compat::WasmCompatSend {
        let model = self.inner.clone();
        async move { model.completion(request).await }
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + rig::wasm_compat::WasmCompatSend {
        async {
            Err(CompletionError::ResponseError(
                "Streaming is not supported by rig_dyn::CompletionModel adapter".to_string(),
            ))
        }
    }
}

#[async_trait]
impl<M> CompletionModel for M
where
    M: completion::CompletionModel + Send + Sync,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + 'static,
{
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        self.completion(request)
            .await
            .map(|response| CompletionResponse {
                choice: response.choice,
                usage: response.usage,
                raw_response: (),
                message_id: response.message_id,
            })
    }
}
