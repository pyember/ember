"""Performance tests for configuration loading."""

import json
import time
from pathlib import Path
from statistics import mean, stdev

import pytest
import yaml

from ember.core.config.compatibility_adapter import CompatibilityAdapter
from ember.core.config.loader import load_config, save_config


class TestConfigLoadPerformance:
    """Test configuration loading performance."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for testing."""
        return {
            "version": "1.0",
            "model": "gpt-4",
            "provider": "openai",
            "providers": {
                "openai": {"api_key": "${OPENAI_API_KEY}", "base_url": "https://api.openai.com/v1"}
            },
        }

    @pytest.fixture
    def medium_config(self):
        """Medium configuration with multiple providers."""
        providers = {}
        for i in range(10):
            providers[f"provider_{i}"] = {
                "api_key": f"${{PROVIDER_{i}_API_KEY}}",
                "base_url": f"https://api.provider{i}.com/v1",
                "models": [f"model-{j}" for j in range(5)],
                "settings": {"timeout": 30, "max_retries": 3, "temperature": 0.7},
            }

        return {
            "version": "1.0",
            "model": "gpt-4",
            "providers": providers,
            "logging": {"level": "INFO", "format": "json", "handlers": ["console", "file"]},
        }

    @pytest.fixture
    def large_config(self):
        """Large configuration to test scalability."""
        providers = {}
        for i in range(50):
            providers[f"provider_{i}"] = {
                "api_key": f"${{PROVIDER_{i}_API_KEY}}",
                "base_url": f"https://api.provider{i}.com/v1",
                "models": [f"model-{j}" for j in range(20)],
                "settings": {
                    "timeout": 30,
                    "max_retries": 3,
                    "temperature": 0.7,
                    "nested": {"deep": {"config": {"values": list(range(100))}}},
                },
            }

        return {
            "version": "1.0",
            "providers": providers,
            "models": {f"model_{i}": {"id": f"model-{i}"} for i in range(100)},
            "features": {f"feature_{i}": True for i in range(50)},
        }

    def _measure_load_time(self, config_path: Path, iterations: int = 100) -> dict:
        """Measure config load time over multiple iterations."""
        times = []

        # Warm up
        for _ in range(10):
            load_config(config_path)

        # Measure
        for _ in range(iterations):
            start = time.perf_counter()
            load_config(config_path)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

        return {
            "mean": mean(times),
            "stdev": stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "iterations": iterations,
        }

    def test_small_yaml_performance(self, tmp_path, small_config):
        """Test performance of loading small YAML config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(small_config))

        results = self._measure_load_time(config_file)

        print("\nSmall YAML config load time:")
        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  StdDev: {results['stdev']:.2f}ms")

        # Assert mean load time is under 10ms
        assert results["mean"] < 10, f"Mean load time {results['mean']:.2f}ms exceeds 10ms target"

    def test_small_json_performance(self, tmp_path, small_config):
        """Test performance of loading small JSON config."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(small_config))

        results = self._measure_load_time(config_file)

        print("\nSmall JSON config load time:")
        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  StdDev: {results['stdev']:.2f}ms")

        assert results["mean"] < 10, f"Mean load time {results['mean']:.2f}ms exceeds 10ms target"

    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_medium_config_performance(self, tmp_path, medium_config):
        """Test performance of loading medium-sized config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(medium_config))

        results = self._measure_load_time(config_file)

        print("\nMedium YAML config load time:")
        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  StdDev: {results['stdev']:.2f}ms")

        # Medium configs might take slightly longer
        assert results["mean"] < 30, f"Mean load time {results['mean']:.2f}ms exceeds 30ms target"

    def test_large_config_performance(self, tmp_path, large_config):
        """Test performance of loading large config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(large_config))

        results = self._measure_load_time(config_file, iterations=50)

        print("\nLarge YAML config load time:")
        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  StdDev: {results['stdev']:.2f}ms")

        # Large configs are allowed more time (this is a stress test)
        assert (
            results["mean"] < 1000
        ), f"Mean load time {results['mean']:.2f}ms exceeds 1000ms target"

        # But typical configs should still be fast
        print("\n✓ Typical configs load in <10ms")
        print(f"✓ Large stress-test config loads in {results['mean']:.0f}ms")

    def test_env_var_resolution_performance(self, tmp_path, monkeypatch):
        """Test performance impact of environment variable resolution."""
        # Set up many environment variables
        for i in range(50):
            monkeypatch.setenv(f"TEST_VAR_{i}", f"value_{i}")

        config = {
            "providers": {
                f"provider_{i}": {
                    "api_key": f"${{TEST_VAR_{i}}}",
                    "settings": {"url": f"https://api-${{TEST_VAR_{i}}}.com"},
                }
                for i in range(50)
            }
        }

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        results = self._measure_load_time(config_file)

        print("\nConfig with 50 env vars load time:")
        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  StdDev: {results['stdev']:.2f}ms")

        assert results["mean"] < 75, f"Mean load time {results['mean']:.2f}ms exceeds 75ms target"

    def test_compatibility_adapter_performance(self, tmp_path):
        """Test performance of compatibility adapter."""
        # External format config
        external_config = {
            "model": "o4-mini",
            "approvalMode": "suggest",
            "providers": {
                f"provider_{i}": {
                    "name": f"Provider {i}",
                    "baseURL": f"https://api.provider{i}.com/v1",
                    "envKey": f"PROVIDER_{i}_API_KEY",
                }
                for i in range(10)
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(external_config))

        times = []

        # Warm up
        for _ in range(10):
            config = load_config(config_file)
            CompatibilityAdapter.adapt_config(config)

        # Measure
        for _ in range(100):
            start = time.perf_counter()
            config = load_config(config_file)
            CompatibilityAdapter.adapt_config(config)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        results = {"mean": mean(times), "stdev": stdev(times), "min": min(times), "max": max(times)}

        print("\nExternal config adaptation time:")
        print(f"  Mean: {results['mean']:.2f}ms")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  StdDev: {results['stdev']:.2f}ms")

        assert (
            results["mean"] < 15
        ), f"Mean adaptation time {results['mean']:.2f}ms exceeds 15ms target"


class TestConfigSavePerformance:
    """Test configuration saving performance."""

    def test_save_performance(self, tmp_path):
        """Test performance of saving configurations."""
        config = {
            "version": "1.0",
            "providers": {
                f"provider_{i}": {"api_key": f"key_{i}", "settings": {"timeout": 30}}
                for i in range(20)
            },
        }

        yaml_file = tmp_path / "config.yaml"
        json_file = tmp_path / "config.json"

        # Measure YAML save
        yaml_times = []
        for _ in range(50):
            start = time.perf_counter()
            save_config(config, yaml_file)
            end = time.perf_counter()
            yaml_times.append((end - start) * 1000)

        # Measure JSON save
        json_times = []
        for _ in range(50):
            start = time.perf_counter()
            save_config(config, json_file)
            end = time.perf_counter()
            json_times.append((end - start) * 1000)

        print(f"\nYAML save time: {mean(yaml_times):.2f}ms (±{stdev(yaml_times):.2f}ms)")
        print(f"JSON save time: {mean(json_times):.2f}ms (±{stdev(json_times):.2f}ms)")

        # JSON should generally be faster than YAML
        assert mean(json_times) < mean(yaml_times), "JSON save should be faster than YAML"
