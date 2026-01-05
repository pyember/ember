import pytest

from ember.models.providers.params import (
    ReasoningEffort,
    ReasoningParams,
    SamplingParams,
    UnifiedChatRequest,
    normalize_chat_request,
    normalize_unified_chat_request,
    split_system_messages,
)
from ember.models.providers.sampling import SamplingFieldMap, apply_sampling_params


def test_normalize_chat_request_coerces_sampling_and_reasoning():
    request = normalize_chat_request(
        model="gpt-4o",
        prompt="hello",
        system=None,
        context=None,
        stream=False,
        raw_kwargs={
            "temperature": "0.3",
            "top_p": 0.9,
            "top_k": "5",
            "presence_penalty": "1.5",
            "frequency_penalty": 0.2,
            "seed": "42",
            "max_tokens": "64",
            "stop": ["END", "END", "STOP"],
            "reasoning": {"effort": "medium", "summary": "auto"},
        },
    )

    sampling = request.sampling
    assert sampling.temperature == pytest.approx(0.3)
    assert sampling.top_p == pytest.approx(0.9)
    assert sampling.top_k == 5
    assert sampling.max_output_tokens == 64
    assert sampling.presence_penalty == pytest.approx(1.5)
    assert sampling.frequency_penalty == pytest.approx(0.2)
    assert sampling.seed == 42
    assert sampling.stop_sequences == ("END", "STOP")
    assert request.extra == {}

    assert request.reasoning is not None
    assert request.reasoning.effort == ReasoningEffort.MEDIUM
    assert request.reasoning.summary == "auto"


def test_apply_sampling_params_maps_fields():
    sampling = SamplingParams(
        temperature=0.2,
        top_p=0.8,
        top_k=4,
        max_output_tokens=128,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        seed=7,
        stop_sequences=("END",),
    )
    mapping = SamplingFieldMap(
        temperature="temp",
        top_p="p",
        top_k="k",
        max_output_tokens="max_tokens",
        presence_penalty="presence",
        frequency_penalty="freq",
        seed="seed",
        stop_sequences="stop",
    )
    target: dict[str, object] = {"existing": True}

    apply_sampling_params(sampling, target, mapping)

    assert target == {
        "existing": True,
        "temp": 0.2,
        "p": 0.8,
        "k": 4,
        "max_tokens": 128,
        "presence": 0.1,
        "freq": 0.2,
        "seed": 7,
        "stop": ["END"],
    }


def test_normalize_unified_chat_request_splits_system_and_prompt():
    messages = (
        {"role": "system", "content": "behave"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "question"},
    )
    sampling = SamplingParams(top_p=0.5, stop_sequences=("STOP",))
    reasoning = ReasoningParams(effort=ReasoningEffort.LOW)

    unified = UnifiedChatRequest(
        provider="openai",
        model="gpt-4o",
        messages=messages,
        sampling=sampling,
        reasoning=reasoning,
    )

    normalized = normalize_unified_chat_request(unified)

    assert normalized.model == "gpt-4o"
    assert normalized.prompt == "question"
    assert normalized.system == "behave"
    assert normalized.context == ({"role": "assistant", "content": "hi"},)
    assert normalized.sampling.top_p == pytest.approx(0.5)
    assert normalized.sampling.stop_sequences == ("STOP",)
    assert normalized.reasoning is reasoning


def test_split_system_messages_requires_role() -> None:
    with pytest.raises(TypeError):
        split_system_messages(({"content": "missing role"},))
