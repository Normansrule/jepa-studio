"""Safe file access for the backend under folders that sandboxed code can write to.

Sandboxed code can put symlinks, FIFOs, sockets and hard links anywhere in its workspace, at any
time between two tool calls. The backend process runs with your full user rights and no sandbox,
so a plain `open(ws / "media/.x/speech.txt", "w")` that follows a planted link writes wherever the
link points (your ~/.bashrc). Every backend read and write under a workspace therefore goes
through this module:

  * the path is walked one component at a time from the root with O_DIRECTORY | O_NOFOLLOW
    (openat-style, `dir_fd`), so no component may be a symlink; missing folders are created
    the same way;
  * the final file is opened with O_NOFOLLOW | O_NONBLOCK and must be a regular file (never a
    FIFO, socket or device, so nothing can block the backend);
  * writes go to a fresh temporary file (O_CREAT | O_EXCL) in the same folder, renamed over the
    target relative to the folder's descriptor: a link at the target is replaced, never followed;
  * JEV's own scratch files (code snippets, speech text, render work folders, caches) live in
    backend-owned storage outside the workspace (jev-data/scratch, or a private temp folder),
    reached with the same helpers.

tests/test_jev_security_round2.py::test_n1_no_raw_file_io_on_workspace_paths fails when code in jev/
calls open(), Path.open/read_*/write_*, mkdir, unlink, PIL/numpy/matplotlib savers and similar on
a path without going through here (or carrying a `# wsfs-ok: <reason>` comment that says why the
path is not under a workspace).

On Windows (no dir_fd) each component is checked with lstat for links and reparse points before
opening; that leaves a race window, and Windows counts as a weak sandbox (every run asks).
"""
from __future__ import annotations

import contextlib
import contextvars
import errno
import os
import secrets
import shutil
import stat
import tempfile
import threading
from pathlib import Path

from .tools.base import NotRegularFile, ToolError

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_BINARY = getattr(os, "O_BINARY", 0)
HAVE_DIRFD = os.open in getattr(os, "supports_dir_fd", set()) and bool(_NOFOLLOW) and bool(_DIRECTORY)


class UnsafePath(ToolError):
    """A path under a workspace goes through a symlink, escapes the root, or is not a regular file."""


# ------------------------------------------------------------------------------------------ paths
def _parts(root: Path, path) -> list[str]:
    """`path` (absolute under `root`, or relative to it) as plain components, never '..'."""
    p = Path(path)
    if p.is_absolute():
        try:
            rel = p.relative_to(root)
        except ValueError:
            # the caller may have resolved a path whose root is itself a symlink (e.g. /tmp -> /private/tmp)
            try:
                rel = p.relative_to(root.resolve())
            except ValueError as e:
                raise UnsafePath(f"{p} is outside {root}") from e
    else:
        rel = p
    parts = [x for x in rel.parts if x not in ("", ".")]
    if any(x == ".." for x in parts):
        raise UnsafePath(f"{path} contains '..'")
    return parts


def _root(root) -> Path:
    r = Path(root)
    if not r.is_absolute():
        r = r.resolve()
    return r


def _open_dir(root: Path, parts: list[str], create: bool) -> int:
    """A descriptor for root/parts..., walking without following any link (mkdir on demand)."""
    fd = os.open(root, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
    try:
        for name in parts:
            try:
                nfd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(name, 0o777, dir_fd=fd)
                except FileExistsError:
                    pass
                nfd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=fd)
            except OSError as e:
                if e.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
                    raise UnsafePath(f"{'/'.join(parts)}: {name!r} is a link or not a folder; JEV does not "
                                     "follow links in the workspace") from e
                raise
            os.close(fd)
            fd = nfd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_components_win(root: Path, parts: list[str], create: bool) -> Path:
    cur = root
    for name in parts:
        cur = cur / name
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            if not create:
                raise
            cur.mkdir()  # wsfs-ok: Windows fallback, each component lstat-checked
            continue
        if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400 or not stat.S_ISDIR(st.st_mode):
            raise UnsafePath(f"{cur} is a link or not a folder")
    return cur


def _regular(fd: int, name: str) -> None:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise NotRegularFile(f"{name} is not a regular file (FIFO, socket, device or folder); JEV only reads and "
                             "writes regular files")


def _final_fd(dfd: int, name: str, flags: int, mode: int = 0o666) -> int:
    try:
        fd = os.open(name, flags | _NOFOLLOW | _NONBLOCK | _CLOEXEC | _BINARY, mode, dir_fd=dfd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENXIO, errno.EISDIR, errno.EWOULDBLOCK, errno.EMLINK):
            raise NotRegularFile(f"{name} is not a regular file (a link, FIFO, socket, device or folder); JEV "
                                 "only reads and writes regular files") from e
        raise
    _regular(fd, name)
    if _NONBLOCK:
        os.set_blocking(fd, True)
    return fd


# ------------------------------------------------------------------------------------------ open / read / write
def open_file(root, path, mode: str = "rb", *, mkdirs: bool = False, encoding: str | None = None,
              newline: str | None = None):
    """open() for a file under `root` without following any link. Modes: r, rb, w, wb, a, ab, x, xb.
    "w" writes in place after the regular-file check (use write_bytes/write_text for atomic writes)."""
    root = _root(root)
    parts = _parts(root, path)
    if not parts:
        raise UnsafePath(f"{path} is the root folder, not a file")
    m = mode.replace("b", "").replace("t", "")
    flags = {"r": os.O_RDONLY, "w": os.O_WRONLY | os.O_CREAT, "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
             "x": os.O_WRONLY | os.O_CREAT | os.O_EXCL}[m]
    if HAVE_DIRFD:
        dfd = _open_dir(root, parts[:-1], mkdirs and m != "r")
        try:
            fd = _final_fd(dfd, parts[-1], flags)
        finally:
            os.close(dfd)
    else:
        parent = _check_components_win(root, parts[:-1], mkdirs and m != "r")
        from .tools.base import _regular_fd
        fd = _regular_fd(parent / parts[-1], flags)
    if m == "w":
        os.ftruncate(fd, 0)
    if "b" in mode:
        return os.fdopen(fd, mode.replace("x", "w"))
    return os.fdopen(fd, mode.replace("x", "w"), encoding=encoding or "utf-8", newline=newline)


def read_bytes(root, path, limit: int | None = None) -> bytes:
    with open_file(root, path, "rb") as f:
        return f.read() if limit is None else f.read(limit)


def read_text(root, path, encoding: str = "utf-8") -> str:
    return read_bytes(root, path).decode(encoding)


class _AtomicWriter:
    """A temporary file next to the target, renamed over it on close (relative to the folder fd)."""

    def __init__(self, root: Path, parts: list[str], binary: bool, encoding: str | None, newline: str | None):
        self.parts = parts
        self.dfd = _open_dir(root, parts[:-1], True) if HAVE_DIRFD else None
        self.parent = None if HAVE_DIRFD else _check_components_win(root, parts[:-1], True)
        self.tmp = f".{parts[-1][:80]}.jev-{secrets.token_hex(6)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC | _BINARY
        fd = os.open(self.tmp, flags, 0o666, dir_fd=self.dfd) if HAVE_DIRFD else os.open(self.parent / self.tmp,
                                                                                         flags, 0o666)
        self.f = os.fdopen(fd, "wb") if binary else os.fdopen(fd, "w", encoding=encoding or "utf-8", newline=newline)

    def __enter__(self):
        return self.f

    def __exit__(self, et, ev, tb):
        try:
            self.f.close()
            if et is None:
                if HAVE_DIRFD:
                    os.replace(self.tmp, self.parts[-1], src_dir_fd=self.dfd, dst_dir_fd=self.dfd)
                else:
                    os.replace(self.parent / self.tmp, self.parent / self.parts[-1])
            else:
                self._drop()
        except BaseException:
            self._drop()
            raise
        finally:
            if self.dfd is not None:
                os.close(self.dfd)
        return False

    def _drop(self):
        try:
            if HAVE_DIRFD:
                os.unlink(self.tmp, dir_fd=self.dfd)
            else:
                os.unlink(self.parent / self.tmp)
        except OSError:
            pass


def atomic_writer(root, path, binary: bool = True, encoding: str | None = None, newline: str | None = None):
    """`with atomic_writer(ws, "a/b.png") as f: ...`: write a new file (parents created, no link
    followed); a link or file already at the target is replaced, never written through."""
    root = _root(root)
    parts = _parts(root, path)
    if not parts:
        raise UnsafePath(f"{path} is the root folder, not a file")
    return _AtomicWriter(root, parts, binary, encoding, newline)


def write_bytes(root, path, data: bytes) -> Path:
    with atomic_writer(root, path, True) as f:
        f.write(data)
    return _root(root).joinpath(*_parts(_root(root), path))


def write_text(root, path, text: str, encoding: str = "utf-8") -> Path:
    return write_bytes(root, path, text.encode(encoding))


def publish(root, path, src) -> Path:
    """Copy a backend-owned file (scratch) to root/path atomically, without following links."""
    with open(src, "rb") as fi, atomic_writer(root, path, True) as fo:  # wsfs-ok: src is backend scratch
        shutil.copyfileobj(fi, fo, 1 << 20)
    return _root(root).joinpath(*_parts(_root(root), path))


def save_image(im, root, path, format: str = "PNG", **kw) -> Path:
    """PIL Image.save into root/path through atomic_writer."""
    with atomic_writer(root, path, True) as f:
        im.save(f, format=format, **kw)
    return _root(root).joinpath(*_parts(_root(root), path))


# ------------------------------------------------------------------------------------------ folders
def makedirs(root, path) -> Path:
    """mkdir -p under root without following links; returns the (lexical) absolute path."""
    root = _root(root)
    parts = _parts(root, path)
    if HAVE_DIRFD:
        os.close(_open_dir(root, parts, True))
    else:
        _check_components_win(root, parts, True)
    return root.joinpath(*parts)


def check_dir(root, path) -> Path:
    """Raise unless root/path is a real folder reached without links (it must exist)."""
    root = _root(root)
    parts = _parts(root, path)
    if HAVE_DIRFD:
        os.close(_open_dir(root, parts, False))
    else:
        _check_components_win(root, parts, False)
    return root.joinpath(*parts)


def ensure_dir(root, path) -> Path:
    """Like makedirs, but a link (or file) planted at the final component is removed first."""
    root = _root(root)
    parts = _parts(root, path)
    if parts:
        parent = makedirs(root, Path(*parts[:-1])) if len(parts) > 1 else root
        p = parent / parts[-1]
        try:
            st = os.lstat(p)
            if not stat.S_ISDIR(st.st_mode):
                unlink(root, Path(*parts))
        except FileNotFoundError:
            pass
    return makedirs(root, path)


def check_output(root, path) -> Path:
    """For a file an external program (in the sandbox) will write: the folders are created without
    links and the target, if present, is a regular file (a link or special file is removed)."""
    root = _root(root)
    parts = _parts(root, path)
    if not parts:
        raise UnsafePath(f"{path} is the root folder")
    makedirs(root, Path(*parts[:-1])) if len(parts) > 1 else None
    full = root.joinpath(*parts)
    try:
        st = os.lstat(full)
        if not stat.S_ISREG(st.st_mode):
            if stat.S_ISDIR(st.st_mode):
                raise UnsafePath(f"{full.name} is a folder")
            unlink(root, full)
    except FileNotFoundError:
        pass
    return full


def unlink(root, path, missing_ok: bool = True) -> None:
    root = _root(root)
    parts = _parts(root, path)
    if not parts:
        return
    try:
        if HAVE_DIRFD:
            dfd = _open_dir(root, parts[:-1], False)
            try:
                os.unlink(parts[-1], dir_fd=dfd)
            finally:
                os.close(dfd)
        else:
            os.unlink(_check_components_win(root, parts[:-1], False) / parts[-1])
    except FileNotFoundError:
        if not missing_ok:
            raise


def rmtree(root, path) -> None:
    """Remove root/path recursively; never follows a link (shutil.rmtree's fd-based walk)."""
    root = _root(root)
    parts = _parts(root, path)
    if not parts:
        raise UnsafePath("refusing to remove the root folder itself")
    try:
        if HAVE_DIRFD and shutil.rmtree.avoids_symlink_attacks:
            dfd = _open_dir(root, parts[:-1], False)
            try:
                st = os.lstat(parts[-1], dir_fd=dfd)
                if stat.S_ISDIR(st.st_mode):
                    shutil.rmtree(parts[-1], dir_fd=dfd)
                else:
                    os.unlink(parts[-1], dir_fd=dfd)
            finally:
                os.close(dfd)
        else:
            target = _check_components_win(root, parts[:-1], False) / parts[-1]
            if target.is_symlink() or not target.is_dir():
                target.unlink()  # wsfs-ok: Windows fallback, parent components lstat-checked
            else:
                shutil.rmtree(target)  # wsfs-ok: Windows fallback
    except FileNotFoundError:
        pass


def is_regular(root, path) -> bool:
    try:
        with open_file(root, path, "rb"):
            return True
    except (OSError, ToolError):
        return False


# ------------------------------------------------------------------------------------------ scratch
_SCRATCH_BASE: contextvars.ContextVar = contextvars.ContextVar("jev_scratch_base", default=None)
_SCRATCH_DIRS: contextvars.ContextVar = contextvars.ContextVar("jev_scratch_dirs", default=None)
_DEFAULT: dict = {}
_DEFAULT_LOCK = threading.Lock()


def default_scratch_base() -> Path:
    """A private (0700) folder for this backend process, used when no jev-data is known: under your
    cache folder (hidden from bubblewrap, and not under /tmp, which every sandbox replaces)."""
    with _DEFAULT_LOCK:
        if "d" not in _DEFAULT or not Path(_DEFAULT["d"]).is_dir():
            base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jev" / "scratch"
            try:
                base.mkdir(parents=True, exist_ok=True, mode=0o700)  # wsfs-ok: backend-owned cache folder
                _DEFAULT["d"] = Path(tempfile.mkdtemp(prefix=f"p{os.getpid()}-", dir=base))
            except OSError:
                _DEFAULT["d"] = Path(tempfile.mkdtemp(prefix="jev-scratch-"))
        return _DEFAULT["d"]


def cache_dir(name: str) -> Path:
    """A persistent backend-owned folder next to the current scratch base (e.g. the footage feature
    cache), never inside a workspace."""
    base = Path(_SCRATCH_BASE.get() or default_scratch_base())
    d = base / name
    d.mkdir(parents=True, exist_ok=True, mode=0o700)  # wsfs-ok: backend-owned cache folder
    return d


def root_of(path, *roots) -> Path:
    """Which of `roots` (or of this block's scratch folders) contains `path`."""
    p = Path(path)
    for r in [*roots, *scratch_roots()]:
        r = Path(r)
        for cand in (r, r.resolve()):
            if p == cand or cand in p.parents:
                return r
    raise UnsafePath(f"{path} is outside the workspace and JEV's scratch folders")


def scratch_base_for(settings: dict | None, workspace) -> Path:
    """jev-data/scratch/<workspace name>: backend-owned, outside the workspace, hidden from sandboxed
    code (bubblewrap and unshare hide jev-data)."""
    from .settings import private_dir
    s = settings or {}
    ws = Path(workspace).resolve()
    data = s.get("data_dir")
    if not data and ws.parent.parent.name == "jev-data":
        data = ws.parent.parent
    if data:
        name = ws.name if ws.name and not ws.name.startswith(".") else "ws"
        return private_dir(Path(data) / "scratch" / name)
    return default_scratch_base()


@contextlib.contextmanager
def use_scratch(base: Path):
    """Scratch folders made with new_scratch() inside this block live under `base` and are removed
    when the block ends."""
    t1 = _SCRATCH_BASE.set(Path(base))
    made: list[Path] = []
    t2 = _SCRATCH_DIRS.set(made)
    try:
        yield Path(base)
    finally:
        _SCRATCH_BASE.reset(t1)
        _SCRATCH_DIRS.reset(t2)
        for d in made:
            shutil.rmtree(d, ignore_errors=True)  # wsfs-ok: backend-owned scratch (fd-based rmtree)


@contextlib.contextmanager
def ensure_scratch(base: Path | None = None):
    """Join the active scratch block, or open one (under `base` or the default) for this call."""
    if _SCRATCH_DIRS.get() is not None:
        yield _SCRATCH_BASE.get()
        return
    with use_scratch(base or _SCRATCH_BASE.get() or default_scratch_base()) as b:
        yield b


def drop_scratch(d: Path) -> None:
    """Remove a scratch folder made with new_scratch() before the block ends."""
    made = _SCRATCH_DIRS.get()
    if made is not None and d in made:
        made.remove(d)
    shutil.rmtree(d, ignore_errors=True)  # wsfs-ok: backend-owned scratch (fd-based rmtree)


def new_scratch(prefix: str = "work-") -> Path:
    """A fresh private folder in backend-owned scratch storage (never inside a workspace)."""
    base = _SCRATCH_BASE.get() or default_scratch_base()
    Path(base).mkdir(parents=True, exist_ok=True, mode=0o700)  # wsfs-ok: backend-owned scratch base
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    made = _SCRATCH_DIRS.get()
    if made is not None:
        made.append(d)
    return d


def scratch_roots() -> list[Path]:
    """Scratch folders created in the current block (media tools mount them into FFmpeg's sandbox)."""
    return list(_SCRATCH_DIRS.get() or [])


def in_scratch(path) -> Path | None:
    """The scratch folder (made in this block) that contains `path`, or None."""
    p = Path(path)
    for d in scratch_roots():
        if p == d or d in p.parents:
            return d
    return None


__all__ = ["UnsafePath", "open_file", "read_bytes", "read_text", "write_bytes", "write_text", "atomic_writer",
           "publish", "save_image", "makedirs", "ensure_dir", "check_dir", "check_output", "unlink", "rmtree",
           "is_regular", "scratch_base_for", "use_scratch", "new_scratch", "scratch_roots", "in_scratch",
           "default_scratch_base", "cache_dir", "root_of", "ensure_scratch", "drop_scratch"]
