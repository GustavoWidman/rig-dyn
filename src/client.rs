use crate::traits::{
    CompletionModel, DynEmbeddingModel, FinalCompletionResponse, RigCompletionModelAdapter,
};

use rig::client::{CompletionClient, EmbeddingsClient};
use rig::completion::{self, CompletionError, CompletionRequest, CompletionResponse};
use rig::providers;
use rig::streaming::StreamingCompletionResponse;

#[derive(Clone)]
pub enum Client {
    Anthropic(providers::anthropic::Client),
    Azure(providers::azure::Client),
    Cohere(providers::cohere::Client),
    DeepSeek(providers::deepseek::Client),
    Gemini(providers::gemini::Client),
    Groq(providers::groq::Client),
    HuggingFace(providers::huggingface::Client),
    Hyperbolic(providers::hyperbolic::Client),
    LlamaFile(providers::llamafile::Client),
    MiniMax(providers::minimax::Client),
    Mira(providers::mira::Client),
    Mistral(providers::mistral::Client),
    Moonshot(providers::moonshot::Client),
    Ollama(providers::ollama::Client),
    OpenAI(providers::openai::Client),
    OpenRouter(providers::openrouter::Client),
    Perplexity(providers::perplexity::Client),
    Together(providers::together::Client),
    // TODO Voyage AI
    Xai(providers::xai::Client),
    // TODO Xiao
    // TODO Z.ai
}

#[derive(Clone)]
pub struct RigClientCompletionModelAdapter {
    client: Client,
    model: String,
}

impl completion::CompletionModel for RigClientCompletionModelAdapter {
    type Response = ();
    type StreamingResponse = FinalCompletionResponse;
    type Client = Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self {
            client: client.clone(),
            model: model.into(),
        }
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + rig::wasm_compat::WasmCompatSend {
        let client = self.client.clone();
        let model = self.model.clone();

        async move {
            let completion_model = Client::completion_model(&client, &model).await;
            completion_model.completion(request).await
        }
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + rig::wasm_compat::WasmCompatSend {
        async {
            Err(CompletionError::ResponseError(
                "Streaming is not supported by rig_dyn::Client adapter".to_string(),
            ))
        }
    }
}

impl CompletionClient for Client {
    type CompletionModel = RigClientCompletionModelAdapter;
}

macro_rules! completion_model {
	($self:expr, $model:expr, {$($variant:ident),*}) => {
		match $self {
			$(
				Client::$variant(client) => Box::new(client.completion_model($model)),
			)*
		}
	}
}

macro_rules! embedding_model {
    ($self:expr, $model:expr, $input_type:expr,
     {$($some_variant:ident),*},
     {$($none_variant:ident),*},
     $cohere_expr:expr) => {
        match $self {
            $(
                Client::$some_variant(client) => Some(Box::new(client.embedding_model($model))),
            )*
            $(
                Client::$none_variant(_) => None,
            )*
            Client::Cohere(client) => $cohere_expr(client),
        }
    }
}

macro_rules! embedding_model_with_ndims {
	($self:expr, $model:expr, $ndims:expr, $input_type:expr,
		{$($some_variant:ident),*},
		{$($none_variant:ident),*},
		$cohere_expr:expr) => {
		   match $self {
				$(
					Client::$some_variant(client) => Some(
						Box::new(client.embedding_model_with_ndims($model, $ndims))
					),
				)*
				$(
					Client::$none_variant(_) => None,
				)*
				Client::Cohere(client) => $cohere_expr(client),
		   }
	   }
}

impl Client {
    /// Returns a completion model wrapper for the given provider and model name.
    pub async fn completion_model(&self, model: &str) -> Box<dyn CompletionModel> {
        completion_model!(
            self, model,
            {
                Anthropic, Azure, Cohere, DeepSeek,
                Gemini, Groq, Hyperbolic, LlamaFile, MiniMax, Mira, Mistral,
                Moonshot, OpenAI, Ollama, Perplexity, Xai,
                HuggingFace, OpenRouter, Together
            }
        )
    }

    /// Returns a completion model compatible with `rig::completion::CompletionModel`.
    pub async fn rig_completion_model(&self, model: &str) -> RigCompletionModelAdapter {
        RigCompletionModelAdapter::from(self.completion_model(model).await)
    }

    /// Returns an embedding model wrapper for the given provider and model name.
    /// Returns `None` if the provider does not support embeddings or
    /// if improper input type is provided (cohere requires a input type).
    pub async fn embedding_model(
        &self,
        model: &str,
        input_type: Option<&str>,
    ) -> Option<Box<dyn DynEmbeddingModel>> {
        embedding_model!(
            self, model, input_type,
            {
                Azure, Gemini, LlamaFile, Mistral, OpenAI, Ollama, Together, OpenRouter
            },
            {
                Anthropic, DeepSeek, Groq, HuggingFace, Hyperbolic, MiniMax, Mira,
                Moonshot, Perplexity, Xai
            },
            |client: &providers::cohere::Client| input_type.map(|input_type| {
                Box::new(
                    client.embedding_model(model, input_type)
                ) as Box<dyn DynEmbeddingModel>
            })
        )
    }

    pub async fn embedding_model_with_ndims(
        &self,
        model: &str,
        ndims: usize,
        input_type: Option<&str>,
    ) -> Option<Box<dyn DynEmbeddingModel>> {
        embedding_model_with_ndims!(
            self, model, ndims, input_type,
            {
                Azure, Gemini, LlamaFile, Mistral, OpenAI, Ollama, Together
            },
            {
                Anthropic, DeepSeek, Groq, HuggingFace, Hyperbolic, MiniMax, Mira,
                Moonshot, Perplexity, OpenRouter, Xai
            },
            |client: &providers::cohere::Client| input_type.map(|input_type| {
                Box::new(
                    client.embedding_model_with_ndims(model, input_type, ndims)
                ) as Box<dyn DynEmbeddingModel>
            })
        )
    }
}
