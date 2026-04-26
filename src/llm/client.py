"""
client.py
---------
Lightweight wrapper around the Ollama REST API for local LLM inference
with Gemma 4 E4B / E2B models.

Provides the ``GemmaClient`` class with:
    - Connection health checking (verify Ollama is running)
    - Automatic RAM-based model selection (E4B / E2B / disabled)
    - Synchronous and streaming text generation
    - Connection retry logic (3 attempts, 2-second exponential backoff)
    - Model warmup for first-load latency hiding
    - Graceful error handling when Ollama is offline

Usage:
    from src.llm.client import GemmaClient
    client = GemmaClient()
    if client.check_health():
        response = client.generate("Analyze this PDF...", system_prompt=SYSTEM_PROMPT)

Note:
    Ollama must be installed and running locally on port 11434.
    The LLM is loaded **on-demand** — it is NOT loaded at app startup to
    conserve RAM on 8GB systems (PRD §11.3, FR-509).
"""

import time
from typing import Generator, Optional

import httpx
import psutil

from src.config import (
    LLM_BASE_URL,
    LLM_FALLBACK_MODEL,
    LLM_MAX_CONTEXT,
    LLM_MODEL,
    RAM_THRESHOLD_MB,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 120.0         # seconds — generous for CPU inference
_WARMUP_TIMEOUT = 60.0           # seconds — first load can be slow
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 2.0          # seconds — exponential backoff base
_HEALTH_TIMEOUT = 5.0            # seconds — quick ping
_MIN_RAM_MB = 3072               # below this → disable LLM entirely
_FALLBACK_RAM_MB = 5120          # below this → use fallback model
_DEFAULT_TEMPERATURE = 0.3       # analytical, low-creativity responses


class GemmaClient:
    """Ollama API client wrapper for Gemma 4 E4B / E2B local inference.

    This client implements the full lifecycle for LLM integration:
    
    1. **Health check** — verify Ollama daemon is reachable
    2. **RAM check** — auto-select E4B, E2B, or disable based on free RAM
    3. **Generate** — synchronous text generation with retry logic
    4. **Stream** — token-by-token streaming for Streamlit UI
    5. **Warmup** — pre-load model into RAM on first use

    Attributes:
        model (str): Active Ollama model identifier (e.g. ``'gemma4:e4b'``).
        base_url (str): Ollama API base URL (default ``http://localhost:11434``).
        max_context (int): Maximum context window tokens.
        is_available (bool): Whether Ollama is reachable and a model is selected.

    Example:
        >>> client = GemmaClient()
        >>> client.check_health()
        True
        >>> response = client.generate("Explain /JS in PDFs")
        >>> print(response)
        '/JS is a JavaScript action ...'
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_context: Optional[int] = None,
    ):
        """Initialize the Gemma client.

        Args:
            model: Ollama model identifier. If None, auto-selects based on
                   available RAM (E4B if ≥5GB, E2B if ≥3GB, disabled otherwise).
            base_url: Ollama REST API base URL. Defaults to config value.
            max_context: Max context window tokens. Defaults to config value.
        """
        self.base_url = base_url or LLM_BASE_URL
        self.max_context = max_context or LLM_MAX_CONTEXT
        self._primary_model = model or LLM_MODEL
        self._fallback_model = LLM_FALLBACK_MODEL
        self.model: Optional[str] = None
        self.is_available: bool = False
        self._warmed_up: bool = False

        logger.info(
            f"GemmaClient initialized — base_url={self.base_url}, "
            f"primary={self._primary_model}, "
            f"fallback={self._fallback_model}, "
            f"max_context={self.max_context}"
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Ping the Ollama API to verify it is running and reachable.

        Sends a GET request to ``/api/tags`` which returns the list of
        locally available models. This is the lightest Ollama endpoint.

        Returns:
            bool: True if Ollama responds within the timeout, False otherwise.
        """
        try:
            with httpx.Client(timeout=_HEALTH_TIMEOUT) as http:
                response = http.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    logger.info(
                        f"Ollama is healthy — {len(models)} models available: "
                        f"{model_names}"
                    )
                    self.is_available = True
                    return True
                else:
                    logger.warning(
                        f"Ollama returned unexpected status {response.status_code}"
                    )
                    self.is_available = False
                    return False
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            logger.warning(
                f"Ollama is not reachable at {self.base_url}: {exc}. "
                f"LLM features will be disabled. "
                f"Start Ollama with: ollama serve"
            )
            self.is_available = False
            return False

    # ------------------------------------------------------------------
    # RAM-based model selection
    # ------------------------------------------------------------------

    def check_ram(self) -> Optional[str]:
        """Auto-select the appropriate LLM model based on available system RAM.

        Implements the PRD §11.3 RAM management strategy:
            - ≥5GB free → ``gemma4:e4b`` (primary, ~3.5GB during inference)
            - ≥3GB and <5GB free → ``gemma4:e2b`` (fallback, ~1.7GB)
            - <3GB free → ``None`` (LLM disabled entirely, log warning)

        Returns:
            str or None: Selected model identifier, or None if insufficient RAM.
        """
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)

        logger.info(
            f"RAM check: {available_mb:.0f} MB available "
            f"({mem.percent}% used, {mem.total / (1024**3):.1f} GB total)"
        )

        if available_mb >= _FALLBACK_RAM_MB:
            self.model = self._primary_model
            logger.info(
                f"Selected primary model: {self.model} "
                f"({available_mb:.0f} MB available ≥ {_FALLBACK_RAM_MB} MB threshold)"
            )
        elif available_mb >= _MIN_RAM_MB:
            self.model = self._fallback_model
            logger.warning(
                f"Low RAM — falling back to lighter model: {self.model} "
                f"({available_mb:.0f} MB available < {_FALLBACK_RAM_MB} MB threshold)"
            )
        else:
            self.model = None
            self.is_available = False
            logger.warning(
                f"Insufficient RAM for LLM — disabled entirely. "
                f"({available_mb:.0f} MB available < {_MIN_RAM_MB} MB minimum). "
                f"Close other applications to free memory."
            )

        return self.model

    # ------------------------------------------------------------------
    # Synchronous generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a complete text response from the LLM.

        Sends a chat-completion request to Ollama with retry logic.
        On failure, retries up to 3 times with exponential backoff.

        Args:
            prompt: User prompt / analysis request.
            system_prompt: System-level instructions (SOC analyst persona).
            temperature: Sampling temperature (0.0–2.0). Default 0.3 for
                         analytical precision.
            max_tokens: Maximum response tokens. Defaults to context limit.

        Returns:
            str: Complete LLM response text.

        Raises:
            ConnectionError: If Ollama is unreachable after all retries.
            RuntimeError: If the model is not selected (call check_ram first).
        """
        if self.model is None:
            raise RuntimeError(
                "No LLM model selected. Call check_ram() first, or ensure "
                "sufficient RAM is available (≥3GB free)."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.max_context,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        last_error = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                logger.info(
                    f"LLM generate (attempt {attempt}/{_RETRY_ATTEMPTS}) — "
                    f"model={self.model}, prompt_len={len(prompt)}"
                )

                with httpx.Client(timeout=_DEFAULT_TIMEOUT) as http:
                    response = http.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")
                    total_duration = data.get("total_duration", 0)
                    eval_count = data.get("eval_count", 0)

                    duration_sec = total_duration / 1e9 if total_duration else 0
                    logger.info(
                        f"LLM response received — {len(content)} chars, "
                        f"{eval_count} tokens, {duration_sec:.1f}s"
                    )
                    return content
                else:
                    last_error = (
                        f"Ollama returned status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    logger.warning(last_error)

            except httpx.TimeoutException:
                last_error = (
                    f"LLM generation timed out after {_DEFAULT_TIMEOUT}s "
                    f"(attempt {attempt})"
                )
                logger.warning(last_error)
            except (httpx.ConnectError, OSError) as exc:
                last_error = f"Connection to Ollama failed: {exc}"
                logger.warning(last_error)

            # Exponential backoff
            if attempt < _RETRY_ATTEMPTS:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay:.0f}s...")
                time.sleep(delay)

        raise ConnectionError(
            f"Failed to generate LLM response after {_RETRY_ATTEMPTS} attempts. "
            f"Last error: {last_error}. "
            f"Ensure Ollama is running: ollama serve"
        )

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> Generator[str, None, None]:
        """Stream LLM response tokens one at a time.

        Designed for the Streamlit UI to display tokens as they arrive,
        providing a responsive user experience during CPU inference.

        Args:
            prompt: User prompt / analysis request.
            system_prompt: System-level instructions.
            temperature: Sampling temperature. Default 0.3.

        Yields:
            str: Individual response tokens/chunks.

        Raises:
            ConnectionError: If Ollama is unreachable.
            RuntimeError: If no model is selected.
        """
        if self.model is None:
            raise RuntimeError(
                "No LLM model selected. Call check_ram() first."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": self.max_context,
            },
        }

        try:
            logger.info(
                f"LLM streaming — model={self.model}, prompt_len={len(prompt)}"
            )

            with httpx.Client(timeout=_DEFAULT_TIMEOUT) as http:
                with http.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        raise ConnectionError(
                            f"Ollama returned status {response.status_code}"
                        )

                    import json as _json
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = _json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                return
                        except _json.JSONDecodeError:
                            continue

        except httpx.TimeoutException:
            logger.error(f"LLM streaming timed out after {_DEFAULT_TIMEOUT}s")
            yield "\n\n⚠️ Response timed out. Try again or use a shorter prompt."
        except (httpx.ConnectError, OSError) as exc:
            logger.error(f"Streaming connection failed: {exc}")
            yield (
                "\n\n⚠️ Could not connect to Ollama. "
                "Ensure it is running: `ollama serve`"
            )

    # ------------------------------------------------------------------
    # Model warmup
    # ------------------------------------------------------------------

    def warmup(self) -> bool:
        """Pre-load the LLM model into RAM by sending a minimal prompt.

        Should be called once when the user first clicks "Analyze with AI"
        to hide the ~5-10 second cold-start latency on CPU. Subsequent
        calls are near-instant.

        Returns:
            bool: True if warmup succeeded, False on failure.
        """
        if self._warmed_up:
            logger.info("Model already warmed up — skipping")
            return True

        if self.model is None:
            logger.warning("Cannot warm up — no model selected")
            return False

        logger.info(f"Warming up model {self.model}...")

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Hello."}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 128,
                    "num_predict": 5,
                },
            }

            with httpx.Client(timeout=_WARMUP_TIMEOUT) as http:
                response = http.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )

            if response.status_code == 200:
                self._warmed_up = True
                logger.info(f"Model {self.model} warmed up successfully")
                return True
            else:
                logger.warning(
                    f"Warmup returned status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False

        except (httpx.TimeoutException, httpx.ConnectError, OSError) as exc:
            logger.warning(f"Warmup failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Convenience: auto-initialize
    # ------------------------------------------------------------------

    def auto_initialize(self) -> bool:
        """Run the full initialization sequence: health → RAM → warmup.

        Convenience method that performs all setup steps in order.

        Returns:
            bool: True if the client is ready for inference.
        """
        if not self.check_health():
            return False

        selected = self.check_ram()
        if selected is None:
            return False

        self.is_available = True
        return True

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "ready" if self.is_available else "offline"
        return (
            f"GemmaClient(model={self.model!r}, "
            f"base_url={self.base_url!r}, status={status})"
        )


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  GemmaClient — Connection Test")
    print("=" * 60)

    client = GemmaClient()

    # Health check
    print(f"\n[1] Health check...")
    healthy = client.check_health()
    print(f"    Ollama reachable: {healthy}")

    # RAM check
    print(f"\n[2] RAM check...")
    selected = client.check_ram()
    print(f"    Selected model: {selected}")

    if healthy and selected:
        # Quick test
        print(f"\n[3] Test generation...")
        try:
            response = client.generate(
                "In one sentence, what is a /JS tag in a PDF file?",
                temperature=0.1,
            )
            print(f"    Response: {response[:200]}")
        except Exception as e:
            print(f"    Generation failed: {e}")
    else:
        print("\n[3] Skipping generation test (Ollama not available)")

    print(f"\n{client}")
    print("=" * 60)
