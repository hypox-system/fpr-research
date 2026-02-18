"""
LLM client abstraction.

Supports: anthropic, ollama, mock (for tests).
"""

import json
import logging
from typing import Any, Dict, Optional

from research.config import get_config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised for LLM API errors."""

    def __init__(self, message: str, provider: str, details: Optional[Dict] = None):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(f"{provider}: {message}")


class LLMClient:
    """
    Provider-agnostic LLM client.

    Supports:
    - anthropic: Claude API via anthropic SDK
    - ollama: Local Ollama API
    - mock: Returns valid HypothesisFX JSON for testing
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider ('anthropic', 'ollama', 'mock').
                      Defaults to config value.
            model: Model name. Defaults to config value.
            **kwargs: Additional config overrides (temperature, max_output_tokens).
        """
        cfg = get_config().get("hypothesis", {})
        self.provider = provider or cfg.get("llm_provider", "mock")
        self.model = model or cfg.get("model", "")
        self.temperature = kwargs.get("temperature", cfg.get("temperature", 0.7))
        self.max_tokens = kwargs.get(
            "max_output_tokens", cfg.get("max_output_tokens", 4000)
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send prompt to LLM and return raw text response.

        Args:
            system_prompt: System/instruction prompt.
            user_prompt: User message with context.

        Returns:
            Raw text response from LLM.

        Raises:
            LLMError: On API errors.
            ValueError: On unknown provider.
        """
        logger.debug(f"Generating with provider={self.provider}, model={self.model}")

        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        elif self.provider == "mock":
            return self._mock_response(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_anthropic(self, system: str, user: str) -> str:
        """Anthropic API via SDK."""
        try:
            import anthropic
        except ImportError:
            raise LLMError(
                "anthropic package not installed. Run: pip install anthropic",
                provider="anthropic",
            )

        try:
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except anthropic.APIError as e:
            raise LLMError(
                str(e),
                provider="anthropic",
                details={"status_code": getattr(e, "status_code", None)},
            )
        except Exception as e:
            raise LLMError(str(e), provider="anthropic")

    def _call_ollama(self, system: str, user: str) -> str:
        """Ollama local API."""
        try:
            import requests
        except ImportError:
            raise LLMError(
                "requests package not installed",
                provider="ollama",
            )

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model or "llama2",
                    "prompt": user,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            raise LLMError(
                str(e),
                provider="ollama",
                details={"url": "http://localhost:11434/api/generate"},
            )

    def _mock_response(self, system: str, user: str) -> str:
        """
        Return a valid HypothesisFX JSON for testing.

        The mock response references sweep_001 and proposes a realistic
        follow-up experiment.
        """
        # Extract sweep_id from user prompt if present, default to sweep_001
        sweep_ref = "sweep_001_ema_pvsra"

        mock_proposal = {
            "experiment_intent": "failure_mitigation",
            "evidence_refs": [sweep_ref],
            "novelty_claim": {
                "coverage_diff": ["longer_ema_periods", "reduced_trade_frequency"],
                "near_dup_score": 0.3,
            },
            "expected_mechanism": (
                "The previous sweep failed due to LOW_SIGNAL with EMA crossovers "
                "generating too many false signals in ranging markets. By using "
                "longer EMA periods (21/55 instead of 8/21), we expect fewer but "
                "higher-quality signals that capture larger trends. The slower "
                "parameters should filter out noise while maintaining the core "
                "trend-following mechanism."
            ),
            "predictions": {
                "trade_duration_median_bars": 25,
                "trades_per_day": 0.8,
                "gross_minus_net_gap": 0.0014,
            },
            "expected_failure_mode": "LOW_SIGNAL",
            "kill_criteria": [
                "OOS Sharpe < -3",
                "Win rate < 25%",
                "Fewer than 50 trades in test period",
            ],
            "compute_budget": {"max_variants": 100, "max_runtime_minutes": 60},
            "sweep_config_yaml": """name: sweep_002_ema_slow
symbol: BTCUSDT
timeframes: ["4h"]
date_range:
  start: "2024-01-01"
  end: "2024-12-31"
sides: ["long"]
entry:
  key: ema_cross
  params:
    fast: [13, 21, 34]
    slow: [55, 89, 144]
exit:
  key: ema_stop_long
  params:
    period: [21, 34]
fee_rate: 0.0006
slippage_rate: 0.0001""",
        }

        return json.dumps(mock_proposal, indent=2)
