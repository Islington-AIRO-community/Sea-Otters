"""EduShield backend -- the same pipeline as the n8n workflow, without n8n.

Free and local: Ollama for the models, RDAP + one HTTP fetch for the live
evidence, a bundled dated register for the rest. No API keys, no paid service.
"""

__all__ = ["app", "core", "pipeline", "tools", "llm", "data", "prompts"]
