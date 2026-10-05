# Contributing

Thanks for helping. The project's rule of thumb is "see it, run it, trust it": every feature
should be visible in the app, runnable from the command line, and backed by a test or a number
someone can check.

## Development setup

```bash
git clone https://github.com/Normansrule/jepa-studio && cd jepa-studio
python -m venv .venv && source .venv/bin/activate
# optional, smaller download on machines without a GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"            # pytest, playwright, ruff
```

### One-shot setup on Ubuntu / WSL2

`scripts/setup_ubuntu.sh` takes a release zip from a fresh Ubuntu terminal to a published repo:
it installs missing apt packages (one `sudo apt-get install`, plus the GitHub CLI), unpacks to
`~/jepa-studio` (an existing clone is updated in place, history kept), builds a conda env or a
`.venv` with PyTorch (CUDA if `nvidia-smi` works, else CPU), runs the tests, then commits, pushes
to `Normansrule/jepa-studio` (SSH host alias if it signs in as Normansrule, else HTTPS through
`gh`), switches Pages to GitHub Actions and optionally pushes a `v*` tag for the installers.

```bash
bash scripts/setup_ubuntu.sh --doctor             # read-only report: Python, conda, git, gh, ssh, network
bash scripts/setup_ubuntu.sh --no-push            # newest jepa-studio*.zip in Downloads -> env + tests
bash scripts/setup_ubuntu.sh --help               # all options (--cpu, --browser-tests, --https, --yes, ...)
```

Everything goes to `~/jepa-studio-setup-<date>.log`; a failing step prints its name, the command
and the log path. `tests/test_setup_script.py` runs the script against stub `sudo`, `apt-get`,
`gh`, `ssh` and `python3` with a local bare repo as the remote; the hooks it uses,
`JEPA_SETUP_SEARCH_DIRS` and `JEPA_SETUP_DRY_PIP=1`, are for tests only.

Node.js (22 or newer) is needed for the browser-engine checks; they are skipped without it.
The desktop shell (`desktop/`) has its own setup, described in [docs/desktop.md](docs/desktop.md).

## Tests

```bash
python -m pytest                    # the whole suite (about half a minute on a laptop CPU)
python -m pytest -m "not slow"      # what CI runs
python -m pytest tests/test_equations_doc.py tests/test_security.py -q
```

* **Node checks.** `tests/js/*.mjs` are small dependency-free Node scripts (parity with PyTorch,
  gradient checks, file writers, a trainer smoke test). Most take arguments, so run them through
  their pytest wrappers, `tests/test_web_engine.py` and `tests/test_world_parity.py`, which call
  `node tests/js/<script>.mjs ...` and compare with Python. `engine_gradcheck.mjs` and
  `trainer_smoke.mjs` can also be run directly with `node`.
* **Browser test.** `tests/test_site_playwright.py` opens the static site in headless Chromium,
  fails on any console error or Content Security Policy violation, clicks through every tab and
  refreshes the README screenshots in `assets/screens/`. It is skipped unless Playwright's
  browser is installed (`python -m playwright install chromium`).
* **Docs are tested too.** `tests/test_equations_doc.py` recomputes every worked example in
  `docs/equations.md`; if you change a formula, update the page and the test together.

## Style

* Python: `ruff check .` (line length 120, configured in `pyproject.toml`). Type hints on public
  functions; docstrings that state the equation or the rule the code implements.
* JavaScript (`site/js/`): plain ES modules, no build step, no third-party runtime dependencies.
  The Content Security Policy forbids inline scripts, inline styles, `eval` and `on*=` attributes;
  insert untrusted text with `textContent`. See [docs/STYLE.md](docs/STYLE.md) for colours,
  typography and chart rules.
* Weights and tensors: safetensors only, never pickle.
* Docs: write acronyms out in full the first time, e.g. "Sketched Isotropic Gaussian
  Regularization (SIGReg)". Only quote results that exist in a file in `runs/` (or that you add
  with the run that produced them), and state the caveats.
* After editing anything in `docs/` or `README.md`, run `python scripts/build_docs_index.py` so the
  in-app assistant searches the new text (CI checks this with `--check`).

## Pull requests

* One topic per pull request; describe what changed and how you checked it.
* New behaviour comes with a test. Security-relevant behaviour also gets a row in
  [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md) naming the file and the test.
* Don't copy code from non-commercial (CC BY-NC) repositories, including the LeJEPA and I-JEPA
  reference implementations; see [docs/references.md](docs/references.md).
* Report security problems privately, not in a pull request: [SECURITY.md](SECURITY.md).

By contributing you agree that your contribution is licensed under the MIT License.
