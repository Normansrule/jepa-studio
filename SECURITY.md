# Security policy

## Reporting a vulnerability

Please report security problems **privately** through GitHub's private vulnerability reporting:
<https://github.com/Normansrule/jepa-studio/security/advisories/new>. Don't open a public issue
for anything exploitable.

Useful to include: what an attacker controls (a weights file, a dataset, a web page, text in a
log...), the steps or a minimal file that reproduces it, what happens, and the version or commit.

This is a small project with one maintainer, so responses are best effort. The aim is to
acknowledge a report within a week, agree on a fix and a disclosure date with you, and credit you
in the advisory unless you prefer otherwise.

## Supported versions

Only the latest release and the `main` branch get security fixes.

## Scope

In scope: the Python package (`jepa_studio/`), the local server (`jepa-studio serve`), the web app
(`site/`), the desktop shell (`desktop/`) and the CI workflows. What the project defends against,
how, and what is **not** enforced yet is written down in
[docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md). Known gaps listed there are still worth reporting
if you find a concrete way to exploit them.

Out of scope: vulnerabilities in PyTorch, Pillow, FFmpeg, safetensors, Tauri or the OS itself
(report those upstream), and attacks that need an attacker already running code as your user.
