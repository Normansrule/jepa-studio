"""scripts/setup_ubuntu.sh, run for real in a sandbox.

Each test gets its own HOME and a PATH whose first folder holds small bash stubs for the commands
that would touch the system or the network (sudo, apt-get, dpkg-query, gh, ssh, curl, nvidia-smi,
python3). The stubs append their arguments to a log the test reads. git is the real git; pushes go
to a local bare repository through `url.<path>.insteadOf` in the sandbox ~/.gitconfig, so the
script's remote URLs (SSH alias or HTTPS) are kept but nothing leaves the machine.

Test-only hooks in the script: JEPA_SETUP_SEARCH_DIRS (where to look for the zip) and
JEPA_SETUP_DRY_PIP=1 (no environment, no pip installs; `python3` from PATH runs the tests).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "setup_ubuntu.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or any(shutil.which(t) is None for t in ("bash", "git", "unzip")),
    reason="the setup script targets Ubuntu: needs Linux with bash, git and unzip",
)

STUBS = {
    "sudo": 'log sudo "$@"; exec "$@"',
    "apt-get": 'log apt-get "$@"; exit "${STUB_APT_RC:-0}"',
    "dpkg-query": r"""log dpkg-query "$@"
pkg="${!#}"
for m in ${STUB_MISSING_PKGS:-}; do [ "$m" = "$pkg" ] && { printf 'unknown ok not-installed'; exit 1; }; done
printf 'install ok installed'""",
    "nvidia-smi": 'log nvidia-smi "$@"; exit 9',
    "curl": 'log curl "$@"; printf 200',
    "python3": r"""log python3 "$@"
case "$*" in
  *"-m pytest"*)
    if [ "${STUB_PYTEST_RC:-0}" = 0 ]; then echo "12 passed in 0.10s"; exit 0; fi
    echo "FAILED tests/test_site_playwright.py::test_tabs - AssertionError"
    echo "1 failed, 11 passed in 0.10s"; exit "$STUB_PYTEST_RC" ;;
  --version) echo "Python 3.12.3" ;;
esac
exit 0""",
    "ssh": r"""log ssh "$@"
if [ "$1" = -G ]; then
  if [ "${STUB_SSH_ALIAS:-0}" = 1 ] && [ "$2" = github-normansrule ]; then echo "hostname github.com"; else echo "hostname $2"; fi
  echo "user git"; exit 0
fi
host="${!#}"
if [ -n "${STUB_SSH_USER:-}" ] && [ "$host" = git@github-normansrule ]; then
  echo "Hi ${STUB_SSH_USER}! You've successfully authenticated, but GitHub does not provide shell access." >&2; exit 1
fi
echo "git@github.com: Permission denied (publickey)." >&2; exit 255""",
    "gh": r"""log gh "$@"
case "$*" in
  --version) echo "gh version 2.80.0 (stub)" ;;
  "auth status"*)
    if [ "${STUB_GH_AUTHED:-1}" = 1 ]; then
      echo "github.com"; echo "  ✓ Logged in to github.com account ${STUB_GH_LOGIN:-Normansrule} (keyring)"; exit 0
    fi
    echo "You are not logged into any GitHub hosts." >&2; exit 1 ;;
  "api user --jq .login") echo "${STUB_GH_LOGIN:-Normansrule}" ;;
  "api -i user") printf 'HTTP/2.0 200 OK\r\nX-Oauth-Scopes: %s\r\n\r\n{}\n' "${STUB_GH_SCOPES-gist, read:org, repo, workflow}" ;;
  "api user/emails"*) echo "${STUB_GH_EMAIL:-aleks@example.com}" ;;
  "api user --jq"*) echo "1+Normansrule@users.noreply.github.com" ;;
  "repo view"*) [ "${STUB_GH_REPO_EXISTS:-0}" = 1 ]; exit $? ;;
  "repo create"*) exit "${STUB_GH_CREATE_RC:-0}" ;;
esac
exit 0""",
}


class Sandbox:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.home = tmp / "home"
        self.home.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.stub_log = tmp / "stub.log"
        self.stub_log.touch()
        self.remote = tmp / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.remote)], check=True)
        for name, body in STUBS.items():
            p = self.bin / name
            p.write_text(
                "#!/usr/bin/env bash\n"
                'log() { printf "%s\\n" "$*" >> "$STUB_LOG"; }\n' + body + "\n"
            )
            p.chmod(0o755)
        (self.home / ".gitconfig").write_text(
            f'[url "{self.remote}"]\n'
            "\tinsteadOf = git@github-normansrule:Normansrule/jepa-studio.git\n"
            "\tinsteadOf = https://github.com/Normansrule/jepa-studio.git\n"
            "[init]\n\tdefaultBranch = main\n"
        )
        self.dest = self.home / "jepa-studio"
        self.downloads = self.home / "Downloads"
        self.downloads.mkdir()

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "HOME": str(self.home),
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "STUB_LOG": str(self.stub_log),
            "JEPA_SETUP_DRY_PIP": "1",
            "JEPA_SETUP_SEARCH_DIRS": str(self.downloads),
            "GIT_CONFIG_NOSYSTEM": "1",
            "LANG": "C.UTF-8",
            "TERM": "dumb",
        }
        env.update(extra)
        return env

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args], env=self.env(**extra), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=180,
        )

    @staticmethod
    def out(r: subprocess.CompletedProcess[str]) -> str:
        """Once the log starts, the script sends stderr through the same tee as stdout."""
        return r.stdout + r.stderr

    def calls(self) -> str:
        return self.stub_log.read_text()

    def remote_refs(self) -> str:
        return subprocess.run(["git", "-C", str(self.remote), "show-ref"], capture_output=True, text=True).stdout


def make_zip(path: Path, *, marker: str = "new", extra: dict[str, str] | None = None, top: str = "jepa-studio") -> Path:
    files = {
        "pyproject.toml": '[project]\nname = "jepa-studio"\n',
        "jepa_studio/__init__.py": '__version__ = "0.2.0"\n',
        "scripts/build_docs_index.py": "print('ok')\n",
        ".github/workflows/pages.yml": "name: Pages\n",
        ".gitignore": ".venv/\n",
        "marker.txt": marker,
        "docs/kept.md": "kept\n",
    }
    files.update(extra or {})
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(f"{top}/{name}" if top else name, text)
    return path


def tree(root: Path) -> dict[str, float]:
    return {str(p.relative_to(root)): p.stat().st_mtime for p in root.rglob("*")}


@pytest.fixture
def sb(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_script_is_valid_bash_and_shellcheck_clean():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    if shutil.which("shellcheck"):
        r = subprocess.run(["shellcheck", "-S", "warning", str(SCRIPT)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout


def test_doctor_reports_and_changes_nothing(sb: Sandbox):
    make_zip(sb.downloads / "jepa-studio-v0.2.0.zip")
    before = tree(sb.home)
    r = sb.run("--doctor")
    assert r.returncode == 0, r.stdout + r.stderr
    for heading in ("OS", "WSL", "Python", "conda", "gh auth", "token scopes", "ssh github-normansrule",
                    "https://pypi.org/simple/", "https://download.pytorch.org/whl/cpu/", "Nothing was changed"):
        assert heading in r.stdout, heading
    assert "jepa-studio-v0.2.0.zip" in r.stdout
    assert tree(sb.home) == before                      # no log file, no repo folder, nothing
    calls = sb.calls()
    lines = calls.splitlines()
    assert not [c for c in lines if c.startswith("apt-get")]
    assert not [c for c in lines if c.startswith("sudo") and c != "sudo -n true"]   # only "do I need a password?"
    assert "repo create" not in calls and "setup-git" not in calls and "auth refresh" not in calls
    assert sb.remote_refs() == ""


def test_zip_discovery_picks_newest_chrome_duplicate(sb: Sandbox):
    win = sb.tmp / "mnt" / "c" / "Users" / "aleks" / "Downloads"
    win.mkdir(parents=True)
    old = make_zip(win / "jepa-studio-v0.2.0.zip", marker="old")
    new = make_zip(win / "jepa-studio-v0.2.0 (1).zip", marker="new")
    other = make_zip(sb.downloads / "jepa-studio-v0.1.0.zip", marker="older")
    now = time.time()
    os.utime(other, (now - 300, now - 300))
    os.utime(old, (now - 200, now - 200))
    os.utime(new, (now - 100, now - 100))
    search = f"{sb.tmp}/mnt/*/Users/*/Downloads:{sb.downloads}"
    r = sb.run("--no-push", "--skip-tests", JEPA_SETUP_SEARCH_DIRS=search)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "jepa-studio-v0.2.0 (1).zip" in r.stdout
    assert (sb.dest / "marker.txt").read_text() == "new"
    assert not (sb.dest / ".git").exists()              # --no-push: no git at all
    logs = list(sb.home.glob("jepa-studio-setup-*.log"))
    assert len(logs) == 1 and "[4/6] Unpack" in logs[0].read_text()
    assert "\x1b[" not in logs[0].read_text()


def test_missing_zip_explains_where_it_looked(sb: Sandbox):
    r = sb.run("--no-push")
    assert r.returncode != 0
    out = sb.out(r)
    assert "no jepa-studio*.zip found" in out and str(sb.downloads) in out
    assert out.count("STOPPED") == 1 and "FAILED at step" not in out


def test_refuses_home_as_destination(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    before = tree(sb.home)
    r = sb.run(str(z), "--dest", str(sb.home), "--no-push")
    assert r.returncode != 0
    assert "Refusing" in r.stderr
    assert tree(sb.home) == before


def test_update_in_place_keeps_history_and_removes_deleted_files(sb: Sandbox):
    sb.dest.mkdir()
    (sb.dest / "gone").mkdir()
    (sb.dest / "gone" / "old_only.txt").write_text("removed upstream")
    (sb.dest / "marker.txt").write_text("old")
    git = ["git", "-C", str(sb.dest), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    env = {**os.environ, "HOME": str(sb.home), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True, env=env)
    subprocess.run([*git, "add", "-A"], check=True, env=env)
    subprocess.run([*git, "commit", "-q", "-m", "first"], check=True, env=env)
    (sb.dest / "my-notes.txt").write_text("untracked, mine")
    z = make_zip(sb.downloads / "jepa-studio.zip", marker="new")
    r = sb.run(str(z), "--no-push", "--skip-tests")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (sb.dest / "gone" / "old_only.txt").exists()
    assert not (sb.dest / "gone").exists()
    assert (sb.dest / "marker.txt").read_text() == "new"
    assert (sb.dest / "my-notes.txt").exists()
    assert (sb.dest / ".git").is_dir()
    log = subprocess.run(["git", "-C", str(sb.dest), "log", "--oneline"], capture_output=True, text=True, env=env)
    assert log.stdout.count("\n") == 1 and "first" in log.stdout


def test_https_path_adds_workflow_scope_and_sets_up_git(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), STUB_GH_SCOPES="gist, read:org, repo")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = sb.calls()
    assert "gh auth refresh -h github.com -s workflow" in calls
    assert "gh auth setup-git -h github.com" in calls
    assert "gh repo create Normansrule/jepa-studio --public" in calls
    assert "gh api -X POST repos/Normansrule/jepa-studio/pages -f build_type=workflow" in calls
    assert "gh workflow run pages.yml -R Normansrule/jepa-studio --ref main" in calls
    cfg = lambda k: subprocess.run(["git", "-C", str(sb.dest), "config", "--local", k],  # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    assert cfg("remote.origin.url") == "https://github.com/Normansrule/jepa-studio.git"
    assert cfg("user.name") == "Aleksander Norman"
    assert cfg("user.email") == "aleks@example.com"
    assert "refs/heads/main" in sb.remote_refs()
    assert "refs/tags/" not in sb.remote_refs()         # no terminal and no --yes: tag question answers no
    assert "Build desktop installers now (push tag v0.2.0)? [y/N] n" in r.stdout


def test_https_path_skips_refresh_when_scope_present(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--skip-tests")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "auth refresh" not in sb.calls()
    assert "gh auth setup-git -h github.com" in sb.calls()


def test_ssh_path_when_alias_signs_in_as_owner(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--yes", STUB_SSH_ALIAS="1", STUB_SSH_USER="Normansrule", STUB_GH_SCOPES="repo")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = sb.calls()
    assert "ssh -T" in calls and "git@github-normansrule" in calls
    assert "setup-git" not in calls and "auth refresh" not in calls
    url = subprocess.run(["git", "-C", str(sb.dest), "config", "remote.origin.url"], capture_output=True, text=True)
    assert url.stdout.strip() == "git@github-normansrule:Normansrule/jepa-studio.git"
    refs = sb.remote_refs()
    assert "refs/heads/main" in refs and "refs/tags/v0.2.0" in refs     # --yes pushes the release tag


def test_ssh_alias_for_other_account_falls_back_to_https(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--skip-tests", STUB_SSH_ALIAS="1", STUB_SSH_USER="N0rmansrule")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "gh auth setup-git" in sb.calls()
    url = subprocess.run(["git", "-C", str(sb.dest), "config", "remote.origin.url"], capture_output=True, text=True)
    assert url.stdout.strip().startswith("https://github.com/")


def test_wrong_gh_account_stops_before_push(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--skip-tests", STUB_GH_LOGIN="N0rmansrule")
    assert r.returncode != 0
    assert "gh auth switch -h github.com -u Normansrule" in sb.out(r)
    assert sb.remote_refs() == ""


def test_failed_tests_without_yes_stop_before_any_push(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), STUB_PYTEST_RC="1")
    assert r.returncode != 0
    assert "Publish anyway? [y/N] n" in r.stdout
    assert "FAILED tests/test_site_playwright.py::test_tabs" in r.stdout
    assert "Nothing was committed or pushed" in sb.out(r)
    assert sb.remote_refs() == ""
    assert "repo create" not in sb.calls() and "auth" not in sb.calls()
    assert not (sb.dest / ".git").exists()
    # browser tests are left out unless --browser-tests
    assert "--ignore=tests/test_site_playwright.py --ignore=tests/test_desktop_tier.py" in sb.calls()


def test_failed_tests_with_yes_publish_anyway(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--yes", STUB_PYTEST_RC="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "refs/heads/main" in sb.remote_refs()


def test_unexpected_failure_names_step_and_log(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--skip-tests", STUB_GH_CREATE_RC="1")
    assert r.returncode != 0
    out = sb.out(r)
    assert "FAILED at step 9 (Push to github.com/Normansrule/jepa-studio)" in out
    assert "command: gh repo create" in out
    assert "Paste the last 40 lines of the log to Claude" in out
    log = next(sb.home.glob("jepa-studio-setup-*.log"))
    assert "FAILED at step 9" in log.read_text()


def test_missing_packages_use_one_apt_install(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--no-push", "--skip-tests", STUB_MISSING_PKGS="python3-venv build-essential")
    assert r.returncode == 0, r.stdout + r.stderr
    installs = [line for line in sb.calls().splitlines() if line.startswith("apt-get") and " install " in line]
    assert len(installs) == 1
    assert installs[0].endswith("install -y -q python3-venv build-essential")


def test_no_apt_when_everything_is_present(sb: Sandbox):
    z = make_zip(sb.downloads / "jepa-studio.zip")
    r = sb.run(str(z), "--no-push", "--skip-tests")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "apt-get" not in sb.calls()
