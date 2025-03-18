"""
Debug script for testing model discovery components directly.
This includes enhanced timeout mechanisms and debug outputs to identify freezing points.
"""

import pytest
import logging
import os
import time
import threading
import signal
import sys
from pprint import pprint
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import discovery components
from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService
from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)


class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass


@contextmanager
def timeout_context(seconds, msg="Execution timed out"):
    """Context manager that raises TimeoutException if the block doesn't finish in time."""
    def timeout_handler(signum, frame):
        raise TimeoutException(msg)
    
    # Register the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def run_with_timeout(func, args=None, kwargs=None, timeout_seconds=30, default_return=None):
    """
    Run a function with a timeout and return the result, or default_return on timeout.
    Also prints diagnostic information about execution time.
    """
    args = args or []
    kwargs = kwargs or {}
    start_time = time.time()
    result = default_return
    
    # Create an event to signal when the thread is done
    done_event = threading.Event()
    thread_result = []
    
    def worker():
        try:
            res = func(*args, **kwargs)
            thread_result.append(res)
        except Exception as e:
            logger.exception(f"Error in worker thread: {e}")
        finally:
            done_event.set()
    
    # Start the worker thread
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    
    # Wait for the thread to complete or timeout
    success = done_event.wait(timeout_seconds)
    duration = time.time() - start_time
    
    if success:
        if thread_result:
            result = thread_result[0]
        logger.info(f"Function {func.__name__} completed in {duration:.2f}s")
    else:
        logger.error(f"Function {func.__name__} timed out after {duration:.2f}s")
        # Attempt thread diagram to help diagnose where it's stuck
        logger.info("Active threads at timeout:")
        for thread in threading.enumerate():
            logger.info(f"  - Thread {thread.name} ({thread.ident}) - Alive: {thread.is_alive()}")
    
    return result, duration, success


def debug_model_discovery_provider(provider_class, provider_name):
    """Run a detailed diagnostic on a single model discovery provider."""
    print(f"\n=== Testing {provider_name} Discovery Provider ===\n")
    
    try:
        # Create the provider with defensive timeout
        provider_instance = provider_class()
        print(f"Successfully created {provider_name} discovery instance")
        
        # Fetch models with timeout
        print(f"Fetching {provider_name} models (timeout: 20s)...")
        models, duration, success = run_with_timeout(
            provider_instance.fetch_models, 
            timeout_seconds=20
        )
        
        if not success:
            print(f"❌ ERROR: {provider_name} model fetch timed out after 20s")
            return False
        
        print(f"✅ Successfully fetched {len(models)} {provider_name} models in {duration:.2f}s")
        
        # Sample models
        print("Sample models:")
        for i, (model_id, model_data) in enumerate(list(models.items())[:3]):
            print(f"  {model_id}: {model_data.get('model_name')}")
        
        # Check if models are from API or fallbacks
        fallback_models = provider_instance._get_fallback_models()
        api_discovered = any(model_id not in fallback_models for model_id in models)
        print(f"Contains non-fallback models: {api_discovered}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR with {provider_name} discovery: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_discovery_service():
    """Run diagnostics on the model discovery service."""
    print("\n=== Testing Model Discovery Service ===\n")
    
    # Create service with diagnostic output
    print("Creating ModelDiscoveryService...")
    discovery_service = ModelDiscoveryService()
    
    # Log initialized providers
    print(f"Service initialized with {len(discovery_service.providers)} providers:")
    for i, provider in enumerate(discovery_service.providers):
        print(f"  {i+1}. {provider.__class__.__name__}")
    
    # Watch for lock issues
    print("Testing discover_models with timeout (30s)...")
    discovered_models, duration, success = run_with_timeout(
        discovery_service.discover_models,
        timeout_seconds=30
    )
    
    if not success:
        print("❌ ERROR: ModelDiscoveryService timed out!")
        print("This is likely due to a hanging provider or lock issue.")
        return False
    
    print(f"✅ Successfully discovered {len(discovered_models)} models in {duration:.2f}s")
    
    # Print discovered models by provider
    openai_ids = [m for m in discovered_models if m.startswith("openai:")]
    anthropic_ids = [m for m in discovered_models if m.startswith("anthropic:")]
    
    print(f"OpenAI models: {len(openai_ids)}")
    print(f"Anthropic models: {len(anthropic_ids)}")
    
    # Sample of models
    if openai_ids:
        print("\nOpenAI models (first 3):")
        for model_id in openai_ids[:3]:
            print(f"  - {model_id}")
    
    if anthropic_ids:
        print("\nAnthropic models (first 3):")
        for model_id in anthropic_ids[:3]:
            print(f"  - {model_id}")
    
    return True


def main():
    """Run a series of debug checks with proper timeouts."""
    print("\n===== EMBER MODEL DISCOVERY DEBUG =====\n")
    print(f"Running at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test OpenAI provider in isolation
    openai_ok = debug_model_discovery_provider(OpenAIDiscovery, "OpenAI")
    
    # Test Anthropic provider in isolation
    anthropic_ok = debug_model_discovery_provider(AnthropicDiscovery, "Anthropic")
    
    # Only test discovery service if at least one provider worked
    if openai_ok or anthropic_ok:
        service_ok = debug_discovery_service()
    else:
        print("\n❌ Skipping discovery service test as all providers failed")
        service_ok = False
    
    # Summary
    print("\n===== DEBUG SUMMARY =====")
    print(f"OpenAI Provider: {'✅ OK' if openai_ok else '❌ Failed'}")
    print(f"Anthropic Provider: {'✅ OK' if anthropic_ok else '❌ Failed'}")
    print(f"Discovery Service: {'✅ OK' if service_ok else '❌ Failed'}")
    
    if not (openai_ok and anthropic_ok and service_ok):
        print("\nIssues detected. Try checking:")
        print("1. Network connectivity to provider APIs")
        print("2. API keys are correctly set in environment")
        print("3. Review logs for timeout or thread issues")
        print("4. Check if the provider is implementing timeout handling correctly")
    else:
        print("\nAll diagnostics passed successfully!")


if __name__ == "__main__":
    main()
