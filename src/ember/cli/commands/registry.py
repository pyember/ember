"""`ember registry` subcommands."""

import argparse

from ember.context import context
from ember.core.credentials import CredentialNotFoundError
from ember.models.catalog import ModelInfo, get_model_info, list_available_models


def cmd_registry_list_models(args: argparse.Namespace) -> int:
    ctx = context.get()
    all_models = ctx.list_models()

    if args.verbose:
        by_provider: dict[str, list[tuple[str, ModelInfo]]] = {}
        for model_id in sorted(all_models):
            try:
                info = get_model_info(model_id, include_dynamic=False)
            except KeyError:
                continue
            provider = info.provider
            by_provider.setdefault(provider, []).append((model_id, info))

        for provider, items in sorted(by_provider.items()):
            print(f"\n{provider.upper()} Models:")
            for model_id, info in items:
                print(f"  {model_id:<25} {info.description}")
                print(f"    Context: {info.context_window:,} tokens")
                if info.capabilities:
                    print("    Capabilities: " + ", ".join(sorted(info.capabilities)))
        return 0

    provider_filter = args.provider
    if provider_filter:
        models = list_available_models(provider=provider_filter, include_dynamic=False)
        print(f"Available {provider_filter} models:")
    else:
        models = all_models
        print("Available models:")

    for model_id in sorted(models):
        print(f"  {model_id}")

    return 0


def cmd_registry_list_providers(args: argparse.Namespace) -> int:
    del args

    ctx = context.get()

    print("Provider Status:")
    for provider in ("openai", "anthropic", "google"):
        try:
            ctx.get_credential(provider)
            status = "Configured"
        except CredentialNotFoundError:
            status = "Not configured"
        print(f"  {provider:<12} {status}")

    return 0


def cmd_registry_info(args: argparse.Namespace) -> int:
    model_id = args.model_id
    try:
        info = get_model_info(model_id, include_dynamic=False)
    except KeyError:
        print(f"Model '{model_id}' not found in catalog")
        return 1

    print(f"Model: {model_id}")
    provider = info.provider
    print(f"Provider: {provider}")
    print(f"Description: {info.description}")

    ctxw = info.context_window
    if ctxw:
        print(f"Context Window: {ctxw:,} tokens")

    from ember.models.pricing import get_model_cost
    from ember.models.pricing.manager import PricingNotFoundError

    try:
        cost = get_model_cost(model_id)
    except PricingNotFoundError as exc:
        print(str(exc))
    else:
        print(f"Cost: ${cost['input']:.4f} input / ${cost['output']:.4f} output per 1M tokens")

    caps = info.capabilities
    if caps:
        print(f"Capabilities: {', '.join(sorted(caps))}")

    return 0
