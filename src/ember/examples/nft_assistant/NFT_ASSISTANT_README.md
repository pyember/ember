# NFT Education Assistant

This project implements an NFT Education Assistant using Ember's model API. The assistant provides personalized explanations of NFT and poker concepts based on the user's expertise level.

## Features

- Uses state-of-the-art language models for accurate explanations
- Adapts explanations based on user expertise level (beginner, intermediate, expert)
- Focuses on NFT and poker-related concepts
- Supports multiple model providers (OpenAI, Anthropic, Google/Deepmind)

## Implementation

The assistant uses Ember's model API to generate personalized explanations based on the user's query and expertise level.

Supported models include:
- OpenAI: GPT-4o, GPT-4o-mini, GPT-4, GPT-4-turbo, GPT-3.5-turbo
- Anthropic: Claude 3 Opus, Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3.7 Sonnet
- Deepmind: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Pro

## Setup

Before using the NFT Education Assistant, you need to set up your API keys for the providers you want to use:

1. Get API keys from the providers you want to use:
   - [OpenAI](https://platform.openai.com/)
   - [Anthropic](https://www.anthropic.com/)
   - [Google AI](https://ai.google.dev/)

2. Set them as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## Usage

```python
from nft_education_assistant import NFTEducationAssistant

# Create an instance of the assistant with a specific model
assistant = NFTEducationAssistant(model_name="openai:gpt-4o")

# Or use a different model provider
# assistant = NFTEducationAssistant(model_name="deepmind:gemini-1.5-pro")
# assistant = NFTEducationAssistant(model_name="anthropic:claude-3-opus")

# Get an explanation for a beginner
beginner_explanation = assistant.explain_concept(
    query="What is a non-fungible token?",
    user_expertise_level="beginner"
)

# Get an explanation for an expert
expert_explanation = assistant.explain_concept(
    query="What are the implications of ERC-721 vs ERC-1155 for poker-based NFTs?",
    user_expertise_level="expert"
)
```

## Running the Test Script

To test the assistant with sample queries (using environment variables for API keys):

```bash
python test_nft_assistant.py
```

Or use the provided script with environment variables already set:

```bash
python run_nft_assistant.py
```

The script will use the Google Gemini model by default, but you can modify it to use a different model by changing the `model_name` parameter in the script.

## Requirements

- Ember framework
- Access to at least one of the following:
  - OpenAI API key (for GPT models)
  - Anthropic API key (for Claude models)
  - Google AI API key (for Gemini models)

## Example Outputs

### Beginner Level Explanation

```
Imagine you have a trading card of your favorite basketball player. It's special, right? It's not exactly the same as any other card, even another card of the same player. It might have a different number, a different condition, or maybe even a unique autograph. You can trade it, sell it, or keep it as part of your collection.

A non-fungible token, or NFT, is kind of like a digital version of that special trading card. "Non-fungible" just means it's unique and can't be replaced with something exactly the same. Think of a dollar bill – that's *fungible*. You can trade one dollar for another, and they're essentially identical. But your trading card, or an NFT, is one-of-a-kind.

NFTs use blockchain technology (think of it like a super-secure digital ledger) to prove ownership and authenticity. So, even though an image or video might be copied easily online, the NFT is like a certificate of ownership saying you own the *original* digital item.
```

### Expert Level Explanation

```
A non-fungible token (NFT) is a cryptographic token representing ownership of a unique digital or physical asset recorded on a distributed ledger, typically a blockchain. It's crucial to understand the nuances beyond the basic definition, particularly at an expert level.

Uniqueness and Indivisibility: NFTs are inherently non-fungible, meaning they are not interchangeable on a 1:1 basis like cryptocurrencies such as Bitcoin or Ethereum. This uniqueness derives from distinct metadata embedded within the token, effectively acting as a digital fingerprint. While an NFT can represent fractional ownership, the token itself remains indivisible, pointing to a specific entry on the blockchain.

Metadata and Provenance: The core value proposition of an NFT often lies in its metadata. This data, structured usually as JSON, defines the asset the NFT represents and can include various information: creator information, creation date, a hash of the underlying asset (image, video, audio, etc.), ownership history, and even embedded smart contract functionality.
```
