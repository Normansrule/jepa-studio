"""The system prompt steers local models to the right tool: physics renders, not diffusion models, for
"render a black hole" (a real qwen2.5-coder run picked generate_image with no model installed)."""
from jev.agent import SYSTEM_PROMPT, system_prompt
from jev.tools.base import load_all


def test_prompt_routes_physics_to_science_tools(tmp_path):
    assert "means render_black_hole" in SYSTEM_PROMPT and "never generate_image" in SYSTEM_PROMPT
    tools = load_all()
    for name in ("generate_image", "generate_video"):
        assert "NOT for black holes" in tools[name].description
    assert tools["render_black_hole"].risk == "sandbox"     # the recommended path runs without asking


def test_prompt_lists_installed_generative_models(tmp_path):
    tools = load_all()
    specs = list(tools.values())
    models = tmp_path / "models"
    p = system_prompt(specs, True, tmp_path, {"media": {"models_dir": str(models)}})
    assert "Generative models installed: none." in p
    (models / "sdxl-turbo").mkdir(parents=True)
    (models / "sdxl-turbo" / "model_index.json").write_text("{}")
    p = system_prompt(specs, False, tmp_path, {"media": {"models_dir": str(models)}})
    assert "Generative models installed: sdxl-turbo." in p
    no_gen = [s for s in specs if not s.name.startswith("generate_")]
    assert "Generative models installed" not in system_prompt(no_gen, True, tmp_path, {})
