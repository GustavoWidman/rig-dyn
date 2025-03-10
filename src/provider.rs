#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use std::fmt::Display;

use rig::providers::{
    anthropic as Anthropic, azure as Azure, cohere as Cohere, deepseek as DeepSeek,
    galadriel as Galadriel, gemini as Gemini, groq as Groq, hyperbolic as Hyperbolic,
    moonshot as Moonshot, ollama as Ollama, openai as OpenAI, perplexity as Perplexity, xai as Xai,
};

use crate::client::Client;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Provider {
    /// Anthropic API
    ///
    /// Alias: `anthropic`
    #[cfg_attr(feature = "serde", serde(rename = "anthropic"))]
    Anthropic,

    /// Azure API
    ///
    /// Alias: `azure`
    #[cfg_attr(feature = "serde", serde(rename = "azure"))]
    Azure,

    /// Cohere API
    ///
    /// Alias: `cohere`
    #[cfg_attr(feature = "serde", serde(rename = "cohere"))]
    Cohere,

    /// Deepseek API
    ///
    /// Alias: `deepseek`
    #[cfg_attr(feature = "serde", serde(rename = "deepseek"))]
    DeepSeek,

    /// Galadriel API
    ///
    /// Alias: `galadriel`
    #[cfg_attr(feature = "serde", serde(rename = "galadriel"))]
    Galadriel,

    /// Gemini API
    ///
    /// Alias: `gemini`
    #[cfg_attr(feature = "serde", serde(rename = "gemini"))]
    Gemini,

    /// Groq API
    ///
    /// Alias: `groq`
    #[cfg_attr(feature = "serde", serde(rename = "groq"))]
    Groq,

    /// Hyperbolic API
    ///
    /// Alias: `hyperbolic`
    #[cfg_attr(feature = "serde", serde(rename = "hyperbolic"))]
    Hyperbolic,

    /// Moonshot API
    ///
    /// Alias: `moonshot`
    #[cfg_attr(feature = "serde", serde(rename = "moonshot"))]
    Moonshot,

    /// OpenAI API
    ///
    /// Alias: `openai`, `openai-api`, `openai-compatible`
    #[cfg_attr(feature = "serde", serde(alias = "openai"))]
    #[cfg_attr(feature = "serde", serde(alias = "openai-api"))]
    #[cfg_attr(feature = "serde", serde(alias = "openai-compatible"))]
    OpenAI,

    /// Ollama API
    ///
    /// Alias: `ollama`
    #[cfg_attr(feature = "serde", serde(rename = "ollama"))]
    Ollama,

    /// Perplexity API
    ///
    /// Alias: `perplexity`
    #[cfg_attr(feature = "serde", serde(rename = "perplexity"))]
    Perplexity,

    /// Xai API
    ///
    /// Alias: `xai`
    #[cfg_attr(feature = "serde", serde(rename = "xai"))]
    Xai,
}

impl Default for Provider {
    fn default() -> Self {
        Self::OpenAI
    }
}

#[cfg(feature = "serde")]
impl TryFrom<String> for Provider {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        serde_plain::from_str(&value).map_err(|e| anyhow::anyhow!("{}", e))
    }
}

#[cfg(feature = "serde")]
impl Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        serde_plain::to_string(self)
            .map_err(|_| std::fmt::Error::default())?
            .fmt(f)
    }
}

macro_rules! provider_client {
	(
		$self:expr, $api_key:expr, $custom_url:expr,
		{$($custom_url_variant:ident),*}, {$($standard_variant:ident),*},
		$azure_expr:expr, $anthropic_expr:expr, $galadriel_expr:expr, $ollama_expr:expr
	) => {
		// get the rig provider module by lowercasing the variant name
		match $self {
            $(
                Provider::$custom_url_variant => match $custom_url {
					None => Client::$custom_url_variant(
						$custom_url_variant::Client::new($api_key)
					),
					Some(url) => Client::$custom_url_variant(
						$custom_url_variant::Client::from_url($api_key, url)
					),
				},
            )*
            $(
                Provider::$standard_variant => Client::$standard_variant(
					$standard_variant::Client::new($api_key)
				),
            )*
			Provider::Anthropic => $anthropic_expr,
			Provider::Azure => $azure_expr
			Provider::Galadriel => $galadriel_expr,
			Provider::Ollama => $ollama_expr,
        }
	}
}

impl Provider {
    pub fn client(&self, api_key: &str, custom_url: Option<&str>) -> anyhow::Result<Client> {
        Ok(provider_client!(self, api_key, custom_url,
            {
                Cohere, DeepSeek, Gemini,
                Groq, Hyperbolic, Moonshot,
                OpenAI, Perplexity
            },
            {
                Xai
            },
            match custom_url {
                Some(url) => {
                    Client::Azure(Azure::Client::new(api_key, "2024-10-21", url))
                }
                None => anyhow::bail!("Azure API requires a custom url"),
            },
            {
                let builder = Anthropic::ClientBuilder::new(api_key);
                if let Some(url) = custom_url {
                    Client::Anthropic(builder.base_url(url).build())
                } else {
                    Client::Anthropic(builder.build())
                }
            },
            match custom_url {
                None => Client::Galadriel(Galadriel::Client::new(api_key, None)),
                Some(url) => {
                    Client::Galadriel(Galadriel::Client::from_url(api_key, url, None))
                }
            },
            match custom_url {
                None => Client::Ollama(Ollama::Client::new()),
                Some(url) => {
                    Client::Ollama(Ollama::Client::from_url(url))
                }
            }
        ))
    }
}
