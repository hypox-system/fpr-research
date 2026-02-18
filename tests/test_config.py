"""
Tests for research/config.py

Verifies:
- Default config loads all sections
- Override via custom path
- Reset clears cache
- Missing file raises error
"""

import os
import tempfile
import pytest
import yaml

from research.config import get_config, get, reset


class TestConfigLoader:
    """Tests for config loading."""

    def setup_method(self):
        """Reset config cache before each test."""
        reset()

    def test_default_config_loads(self):
        """Default config should load all sections."""
        config = get_config()

        assert 'post_mortem' in config
        assert 'verifier' in config
        assert 'hypothesis' in config
        assert 'kb_retrieval' in config

    def test_post_mortem_thresholds(self):
        """Post-mortem section should have thresholds."""
        config = get_config()
        thresholds = config['post_mortem']['thresholds']

        assert 'fee_share_pct' in thresholds
        assert 'low_signal_sharpe' in thresholds
        assert thresholds['fee_share_pct'] == 30

    def test_verifier_values(self):
        """Verifier section should have required values."""
        config = get_config()
        verifier = config['verifier']

        assert verifier['min_evidence_refs'] == 2
        assert verifier['min_evidence_refs_bootstrap'] == 1
        assert verifier['bootstrap_threshold'] == 3
        assert verifier['max_variants'] == 500

    def test_hypothesis_values(self):
        """Hypothesis section should have LLM config."""
        config = get_config()
        hypothesis = config['hypothesis']

        assert 'llm_provider' in hypothesis
        assert 'model' in hypothesis
        assert 'temperature' in hypothesis
        assert 'prompt_path' in hypothesis

    def test_kb_retrieval_values(self):
        """KB retrieval section should have context limits."""
        config = get_config()
        retrieval = config['kb_retrieval']

        assert retrieval['max_context_tokens'] == 2000
        assert retrieval['top_n_findings'] == 10


class TestConfigGet:
    """Tests for get() convenience function."""

    def setup_method(self):
        """Reset config cache before each test."""
        reset()

    def test_get_existing_value(self):
        """Get should return existing value."""
        value = get('verifier', 'max_variants')
        assert value == 500

    def test_get_with_default(self):
        """Get should return default for missing key."""
        value = get('verifier', 'nonexistent_key', 'default_value')
        assert value == 'default_value'

    def test_get_missing_section(self):
        """Get should return default for missing section."""
        value = get('nonexistent_section', 'key', 'default')
        assert value == 'default'


class TestConfigOverride:
    """Tests for config override."""

    def setup_method(self):
        """Reset config cache before each test."""
        reset()

    def test_override_via_path(self):
        """Config should load from custom path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            custom_config = {
                'verifier': {
                    'max_variants': 999,
                    'min_evidence_refs': 5,
                },
                'post_mortem': {},
                'hypothesis': {},
                'kb_retrieval': {},
            }
            yaml.dump(custom_config, f)
            custom_path = f.name

        try:
            config = get_config(custom_path)
            assert config['verifier']['max_variants'] == 999
        finally:
            os.unlink(custom_path)

    def test_override_via_env_var(self):
        """Config should load from FPR_LAB_CONFIG env var."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            custom_config = {
                'verifier': {'max_variants': 777},
                'post_mortem': {},
                'hypothesis': {},
                'kb_retrieval': {},
            }
            yaml.dump(custom_config, f)
            custom_path = f.name

        try:
            # Set env var
            original = os.environ.get('FPR_LAB_CONFIG')
            os.environ['FPR_LAB_CONFIG'] = custom_path
            reset()

            config = get_config()
            assert config['verifier']['max_variants'] == 777
        finally:
            # Restore
            if original:
                os.environ['FPR_LAB_CONFIG'] = original
            else:
                os.environ.pop('FPR_LAB_CONFIG', None)
            os.unlink(custom_path)
            # Reset cache after cleanup to avoid polluting other tests
            reset()


class TestConfigReset:
    """Tests for reset() function."""

    def test_reset_clears_cache(self):
        """Reset should clear cached config."""
        # Load default config
        config1 = get_config()
        assert config1['verifier']['max_variants'] == 500

        # Create custom config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            custom_config = {
                'verifier': {'max_variants': 123},
                'post_mortem': {},
                'hypothesis': {},
                'kb_retrieval': {},
            }
            yaml.dump(custom_config, f)
            custom_path = f.name

        try:
            # Load custom config
            reset()
            config2 = get_config(custom_path)
            assert config2['verifier']['max_variants'] == 123

            # Reset and load default again
            reset()
            config3 = get_config()
            assert config3['verifier']['max_variants'] == 500
        finally:
            os.unlink(custom_path)


class TestConfigErrors:
    """Tests for error handling."""

    def setup_method(self):
        """Reset config cache before each test."""
        reset()

    def test_missing_file_raises_error(self):
        """Missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            get_config('/nonexistent/path/to/config.yaml')
