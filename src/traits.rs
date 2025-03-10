use async_trait::async_trait;
use rig::{
    OneOrMany,
    completion::{CompletionError, CompletionRequest},
    embeddings::{self, Embedding, EmbeddingError},
    message::AssistantContent,
};

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError>;
    async fn embed_texts(&self, input: Vec<String>) -> Result<Vec<Embedding>, EmbeddingError>;
    fn ndims(&self) -> usize;
}

#[async_trait]
impl<T> EmbeddingModel for T
where
    T: embeddings::EmbeddingModel + Send + Sync,
{
    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError> {
        embeddings::EmbeddingModel::embed_text(self, input).await
    }

    async fn embed_texts(&self, input: Vec<String>) -> Result<Vec<Embedding>, EmbeddingError> {
        embeddings::EmbeddingModel::embed_texts(self, input).await
    }

    fn ndims(&self) -> usize {
        embeddings::EmbeddingModel::ndims(self)
    }
}

#[async_trait]
pub trait CompletionModel: Send + Sync {
    async fn completion(
        &self,
        completion: CompletionRequest,
    ) -> Result<OneOrMany<AssistantContent>, CompletionError>;
}

#[async_trait]
impl<M> CompletionModel for M
where
    M: rig::completion::CompletionModel + Send + Sync,
{
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<OneOrMany<AssistantContent>, CompletionError> {
        Ok(self.completion(request).await?.choice)
    }
}
