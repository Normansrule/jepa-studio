"""JEV: a local AI agent that lives inside JEPA Studio.

JEV answers questions, writes and runs code, makes images and videos (narrated explainers,
edits of your own footage, optional local generative models) and renders physics, including an
Interstellar-style black hole. Everything runs on your machine: models come from your local
Ollama, files live in folders you control, and nothing is sent anywhere unless you approve it.

Layout:
    jev.tools.base     the Tool interface every capability implements (contract for all modules)
    jev.policy         sandbox + approval safety net (what runs freely, what must ask)
    jev.llm            local model access (Ollama HTTP API) and the model manager
    jev.agent          the tool-calling loop
    jev.store          SQLite: conversations, tool calls, approvals audit log, artifacts
    jev.api            /api/jev/* routes mounted on the jepa_studio local server
    jev.science        black-hole ray tracer and physics simulations
    jev.media          speech, slides, video assembly, footage editing, generative adapters
"""
__version__ = "0.4.0"
