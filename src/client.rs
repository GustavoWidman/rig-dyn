use crate::traits::{CompletionModel, EmbeddingModel};
use rig::providers;

#[derive(Clone)]
pub enum Client {
    Anthropic(providers::anthropic::Client),
    Azure(providers::azure::Client),
    Cohere(providers::cohere::Client),
    DeepSeek(providers::deepseek::Client),
    Galadriel(providers::galadriel::Client),
    Gemini(providers::gemini::Client),
    Groq(providers::groq::Client),
    HuggingFace(providers::huggingface::Client),
    Hyperbolic(providers::hyperbolic::Client),
    Mira(providers::mira::Client),
    Moonshot(providers::moonshot::Client),
    OpenAI(providers::openai::Client),
    OpenRouter(providers::openrouter::Client),
    Ollama(providers::ollama::Client),
    Perplexity(providers::perplexity::Client),
    Together(providers::together::Client),
    Xai(providers::xai::Client),
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
                Galadriel, Gemini, Groq, Hyperbolic,
                Moonshot, OpenAI, Ollama, Perplexity, Xai,
                HuggingFace, OpenRouter, Mira, Together
            }
        )
    }

    /// Returns an embedding model wrapper for the given provider and model name.
    /// Returns `None` if the provider does not support embeddings or
    /// if improper input type is provided (cohere requires a input type).
    pub async fn embedding_model(
        &self,
        model: &str,
        input_type: Option<&str>,
    ) -> Option<Box<dyn EmbeddingModel>> {
        embedding_model!(
            self, model, input_type,
            {
                Azure, Gemini, OpenAI, Xai, Ollama, Together
            },
            {
                Anthropic, DeepSeek, Galadriel,
                Groq, Hyperbolic, Moonshot, Perplexity,
                Mira, HuggingFace, OpenRouter
            },
            |client: &providers::cohere::Client| input_type.map(|input_type| {
                Box::new(
                    client.embedding_model(model, input_type)
                ) as Box<dyn EmbeddingModel>
            })
        )
    }

    pub async fn embedding_model_with_ndims(
        &self,
        model: &str,
        ndims: usize,
        input_type: Option<&str>,
    ) -> Option<Box<dyn EmbeddingModel>> {
        embedding_model_with_ndims!(
            self, model, ndims, input_type,
            {
                Azure, Gemini, OpenAI, Xai, Ollama, Together
            },
            {
                Anthropic, DeepSeek, Galadriel,
                Groq, Hyperbolic, Moonshot, Perplexity,
                Mira, HuggingFace, OpenRouter
            },
            |client: &providers::cohere::Client| input_type.map(|input_type| {
                Box::new(
                    client.embedding_model_with_ndims(model, input_type, ndims)
                ) as Box<dyn EmbeddingModel>
            })
        )
    }
}
