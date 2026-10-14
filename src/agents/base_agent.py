"""
Abstract agent base class providing unified LLM call interface.

All agents (Doctor, Patient, Exam) inherit from this base class,
which provides OpenAI-compatible LLM calling, image-to-base64
conversion, and permission filter binding.
"""

from abc import ABC, abstractmethod
from typing import Optional, List


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, agent_id: str, message_bus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.permission_filter = None  # Injected by session manager
        self.llm_config = {}           # {model, endpoint, api_key}

    def set_llm_config(self, model_name: str = None, model: str = None,
                       endpoint: str = "", api_key: str = ""):
        """Configure the LLM backend for this agent.

        Args:
            model: Model identifier (e.g., 'qwen3.7-plus')
            endpoint: OpenAI-compatible API endpoint URL
            api_key: API key (configure via environment variable, not hardcoded)
        """
        self.llm_config = {
            "model": model or model_name or "",
            "endpoint": endpoint,
            "api_key": api_key,
        }

    def set_permission(self, permission_filter):
        """Bind the permission filter controlling what data this agent can access."""
        self.permission_filter = permission_filter

    @abstractmethod
    def receive(self, message) -> Optional[any]:
        """Receive a message and return a response. Subclasses must implement."""
        raise NotImplementedError

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  images: List = None, max_tokens: int = 8192,
                  temperature: float = 0.6) -> str:
        """Unified LLM call interface (OpenAI-compatible).

        If image analysis fails, falls back to text-only mode.
        """
        from openai import OpenAI

        client = OpenAI(
            base_url=self.llm_config.get("endpoint", ""),
            api_key=self.llm_config.get("api_key", ""),
        )
        model = self.llm_config.get("model", "")

        def _do_call(msgs):
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=max_tokens, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""

        # Request with images
        if images:
            user_content = []
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_to_base64(img)},
                })
            user_content.append({"type": "text", "text": user_prompt})
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            try:
                result = _do_call(msgs)
                if result:
                    return result
            except Exception:
                pass  # Fall back to text-only

        # Text-only request
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return _do_call(msgs)
        except Exception as e:
            print(f"   LLM call failed ({model[:30]}...): {e}")
            return ""

    @staticmethod
    def _image_to_base64(pil_image) -> str:
        """Convert a PIL image to base64 data URI for multimodal LLM requests."""
        import base64
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
