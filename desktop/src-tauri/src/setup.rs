//! First-launch setup: find a system Python >= 3.10, create a private venv in the app's local data
//! folder, pick a PyTorch wheel index for the detected accelerator, and pip-install the pinned
//! requirements plus the bundled `jepa_studio` package (with `--no-deps`).
//!
//! Everything that decides something is a pure function with unit tests at the bottom; the rest
//! runs processes and reports progress through a callback.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The pinned requirements, compiled into the binary so the stamp in setup.json always matches
/// what this build of the shell expects.
pub const REQUIREMENTS: &str = include_str!("../../../requirements-desktop.txt");

pub const MIN_PYTHON: (u32, u32) = (3, 10);
const PYPI_TORCH: &str = "https://download.pytorch.org/whl";

// ------------------------------------------------------------------ accelerator -> wheel index

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    Mac,
    Linux,
}

impl Os {
    pub fn current() -> Os {
        if cfg!(target_os = "windows") {
            Os::Windows
        } else if cfg!(target_os = "macos") {
            Os::Mac
        } else {
            Os::Linux
        }
    }
}

/// What `nvidia-smi` told us. Both fields are optional because old drivers lack some queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NvidiaInfo {
    /// Highest CUDA version the installed driver supports ("CUDA Version: 12.8" in the banner).
    pub driver_cuda: Option<(u32, u32)>,
    /// Lowest compute capability among the visible GPUs (e.g. (8, 6) for an RTX 3060).
    pub min_compute_cap: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Detected {
    pub nvidia: Option<NvidiaInfo>,
    pub rocm: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TorchChoice {
    /// "nvidia", "rocm", "mps" or "cpu".
    pub accelerator: String,
    /// Wheel indexes to try in order; `None` means the default PyPI index. The last entry is
    /// always a safe fallback (the CPU index, or PyPI on macOS).
    pub indexes: Vec<Option<String>>,
}

fn idx(tag: &str) -> Option<String> {
    Some(format!("{PYPI_TORCH}/{tag}"))
}

/// Pure decision: which PyTorch wheel indexes to try, given the OS and the detected hardware.
///
/// * macOS: default PyPI (arm64 wheels include MPS support).
/// * NVIDIA: CUDA indexes newest first, dropping any the driver cannot run (CUDA 13 needs a
///   580+ driver) or that no longer ship kernels for the GPU (CUDA 13 wheels need compute
///   capability >= 7.5, the cu128 builds >= 7.0), then the CPU index as a last resort.
/// * AMD ROCm (Linux only, `rocminfo` present): ROCm indexes newest first, then CPU.
/// * Otherwise: the CPU index, which avoids pulling gigabytes of CUDA libraries from PyPI.
pub fn torch_choice(os: Os, det: &Detected) -> TorchChoice {
    if os == Os::Mac {
        return TorchChoice { accelerator: "mps".into(), indexes: vec![None] };
    }
    if let Some(nv) = det.nvidia {
        // (tag, minimum driver CUDA version, minimum compute capability)
        type Ver = (u32, u32);
        let table: [(&str, Ver, Ver); 3] =
            [("cu130", (13, 0), (7, 5)), ("cu128", (12, 8), (7, 0)), ("cu126", (12, 6), (5, 0))];
        let mut indexes: Vec<Option<String>> = table
            .iter()
            .filter(|(_, drv, cc)| {
                nv.driver_cuda.is_none_or(|d| d >= *drv) && nv.min_compute_cap.is_none_or(|c| c >= *cc)
            })
            .map(|(tag, _, _)| idx(tag))
            .collect();
        if !indexes.is_empty() {
            indexes.push(idx("cpu"));
            return TorchChoice { accelerator: "nvidia".into(), indexes };
        }
        // Driver too old for any CUDA build we pin against: CPU wheels still work.
        return TorchChoice { accelerator: "cpu".into(), indexes: vec![idx("cpu")] };
    }
    if det.rocm && os == Os::Linux {
        return TorchChoice { accelerator: "rocm".into(), indexes: vec![idx("rocm7.0"), idx("rocm6.4"), idx("cpu")] };
    }
    TorchChoice { accelerator: "cpu".into(), indexes: vec![idx("cpu")] }
}

/// Parse `nvidia-smi` banner output for "CUDA Version: X.Y".
pub fn parse_driver_cuda(banner: &str) -> Option<(u32, u32)> {
    let rest = &banner[banner.find("CUDA Version:")? + "CUDA Version:".len()..];
    parse_major_minor(rest.split_whitespace().next()?)
}

/// Parse `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` (one line per GPU) and return
/// the lowest capability, so the chosen wheels have kernels for every GPU in the machine.
pub fn parse_min_compute_cap(csv: &str) -> Option<(u32, u32)> {
    csv.lines().filter_map(|l| parse_major_minor(l.trim())).min()
}

fn parse_major_minor(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once('.')?;
    let b: String = b.chars().take_while(|c| c.is_ascii_digit()).collect();
    Some((a.trim().parse().ok()?, b.parse().ok()?))
}

/// `torch==X` and `torchvision==Y` lines from the requirements, installed first from the chosen
/// accelerator index.
pub fn torch_pins(requirements: &str) -> Vec<String> {
    requirements
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| {
            let name = l.split(|c: char| "=<>!~ ;[".contains(c)).next().unwrap_or("");
            name.eq_ignore_ascii_case("torch") || name.eq_ignore_ascii_case("torchvision")
        })
        .map(str::to_string)
        .collect()
}

pub fn detect_hardware() -> Detected {
    let mut det = Detected::default();
    if let Ok(out) = quiet(Command::new("nvidia-smi")).output() {
        if out.status.success() {
            let banner = String::from_utf8_lossy(&out.stdout);
            let mut nv = NvidiaInfo { driver_cuda: parse_driver_cuda(&banner), min_compute_cap: None };
            if let Ok(q) =
                quiet(Command::new("nvidia-smi")).args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output()
            {
                if q.status.success() {
                    nv.min_compute_cap = parse_min_compute_cap(&String::from_utf8_lossy(&q.stdout));
                }
            }
            det.nvidia = Some(nv);
        }
    }
    if cfg!(target_os = "linux") {
        det.rocm = quiet(Command::new("rocminfo")).output().map(|o| o.status.success()).unwrap_or(false);
    }
    det
}

// ------------------------------------------------------------------ python discovery

/// Parse the output of `python -c "import sys; print(*sys.version_info[:2])"`.
pub fn parse_python_version(s: &str) -> Option<(u32, u32)> {
    let mut it = s.split_whitespace();
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

/// A Python interpreter invocation: program plus leading args (e.g. `py -3.12`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PythonCmd {
    pub program: String,
    pub args: Vec<String>,
}

impl PythonCmd {
    pub fn command(&self) -> Command {
        let mut c = Command::new(&self.program);
        c.args(&self.args);
        quiet(c)
    }
    pub fn display(&self) -> String {
        std::iter::once(self.program.as_str()).chain(self.args.iter().map(String::as_str)).collect::<Vec<_>>().join(" ")
    }
}

fn candidates(os: Os) -> Vec<PythonCmd> {
    let pc = |p: &str, a: &[&str]| PythonCmd { program: p.into(), args: a.iter().map(|s| s.to_string()).collect() };
    match os {
        Os::Windows => {
            let mut v: Vec<PythonCmd> =
                ["3.13", "3.12", "3.11", "3.10"].iter().map(|m| pc("py", &[&format!("-{m}")])).collect();
            v.push(pc("py", &["-3"]));
            v.push(pc("python", &[]));
            v
        }
        _ => {
            let mut v: Vec<PythonCmd> =
                ["python3.13", "python3.12", "python3.11", "python3.10"].iter().map(|p| pc(p, &[])).collect();
            v.push(pc("python3", &[]));
            // Common install locations that are not always on a GUI app's PATH (macOS apps get a
            // minimal PATH when launched from Finder).
            for p in ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"] {
                v.push(pc(p, &[]));
            }
            v
        }
    }
}

pub fn python_version(py: &PythonCmd) -> Option<(u32, u32)> {
    let out =
        py.command().args(["-c", "import sys; print(*sys.version_info[:2])"]).stdin(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_python_version(&String::from_utf8_lossy(&out.stdout))
}

/// First interpreter >= 3.10, as (command, version).
pub fn find_python() -> Result<(PythonCmd, (u32, u32)), String> {
    let mut seen = Vec::new();
    for c in candidates(Os::current()) {
        if let Some(v) = python_version(&c) {
            if v >= MIN_PYTHON {
                return Ok((c, v));
            }
            seen.push(format!("{} is {}.{}", c.display(), v.0, v.1));
        }
    }
    let found = if seen.is_empty() { "no Python interpreter was found on PATH".to_string() } else { seen.join(", ") };
    Err(format!(
        "jepa-studio needs Python {}.{} or newer to set up its private environment ({found}). \
         Install it from https://www.python.org/downloads/ (on Windows tick \"Add python.exe to PATH\" \
         or keep the py launcher), then start jepa-studio again.",
        MIN_PYTHON.0, MIN_PYTHON.1
    ))
}

pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

// ------------------------------------------------------------------ setup.json

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SetupRecord {
    pub schema: u32,
    pub app_version: String,
    /// sha256 over the requirements file and the app version: a mismatch re-runs the install.
    pub stamp: String,
    pub base_python: String,
    pub python_version: String,
    pub venv_python: PathBuf,
    pub accelerator: String,
    /// The index torch actually came from ("pypi" for the default index).
    pub torch_index: String,
    pub torch_indexes_tried: Vec<String>,
    pub completed_at_unix: u64,
}

pub fn stamp(requirements: &str, app_version: &str) -> String {
    let mut h = Sha256::new();
    h.update(requirements.as_bytes());
    h.update(b"\0");
    h.update(app_version.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn read_record(path: &Path) -> Option<SetupRecord> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

/// A previous setup is reusable when it finished for the same requirements and app version and
/// its interpreter still exists.
pub fn record_is_current(rec: &SetupRecord, stamp: &str) -> bool {
    rec.schema == 1 && rec.stamp == stamp && rec.venv_python.is_file()
}

// ------------------------------------------------------------------ running pip with progress

/// Progress callback: (fraction 0..1 or None, headline, latest output line).
pub type Progress<'a> = &'a (dyn Fn(Option<f64>, &str, &str) + Sync);

pub struct Logger {
    file: Mutex<fs::File>,
}

impl Logger {
    pub fn open(path: &Path) -> std::io::Result<Arc<Logger>> {
        if let Some(d) = path.parent() {
            fs::create_dir_all(d)?;
        }
        let file = fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Arc::new(Logger { file: Mutex::new(file) }))
    }
    pub fn line(&self, s: &str) {
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{s}");
        }
    }
}

/// Run a command, stream stdout lines to `on_line` and the log, keep a tail of stderr for the error.
pub fn run_logged(mut cmd: Command, log: &Logger, on_line: &dyn Fn(&str)) -> Result<(), String> {
    log.line(&format!("$ {cmd:?}"));
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {:?}: {e}", cmd.get_program()))?;
    let stderr = child.stderr.take().expect("piped stderr");
    let tail = Arc::new(Mutex::new(Vec::<String>::new()));
    let tail2 = tail.clone();
    let err_thread = std::thread::spawn(move || {
        let mut lines = Vec::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            lines.push(line);
        }
        *tail2.lock().unwrap() = lines;
    });
    for line in BufReader::new(child.stdout.take().expect("piped stdout")).lines().map_while(Result::ok) {
        log.line(&line);
        on_line(&line);
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    let _ = err_thread.join();
    let err_lines = tail.lock().unwrap().clone();
    for l in &err_lines {
        log.line(l);
    }
    if status.success() {
        Ok(())
    } else {
        // pip's "ERROR:" lines say what went wrong; retries and warnings come before them.
        let errors: Vec<&String> = err_lines.iter().filter(|l| l.starts_with("ERROR")).collect();
        let t = if errors.is_empty() {
            err_lines.iter().rev().take(12).rev().cloned().collect::<Vec<_>>().join("\n")
        } else {
            errors.iter().map(|l| l.as_str()).collect::<Vec<_>>().join("\n")
        };
        Err(format!("{t}\n({:?} exited with {status})", cmd.get_program()))
    }
}

fn pip(py: &Path) -> Command {
    let mut c = quiet(Command::new(py));
    c.args(["-m", "pip", "--disable-pip-version-check", "--no-input"])
        .env("PYTHONUNBUFFERED", "1")
        .env("PIP_PROGRESS_BAR", "off")
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .env_remove("PIP_INDEX_URL")
        .env_remove("PIP_EXTRA_INDEX_URL");
    c
}

/// Copy the bundled package source to a writable staging folder (setuptools writes egg-info next
/// to the sources, and the bundle may live on a read-only volume) with a minimal pyproject.toml.
fn stage_package(bundled_pkg: &Path, staging: &Path, version: &str) -> Result<(), String> {
    let _ = fs::remove_dir_all(staging);
    copy_tree(bundled_pkg, &staging.join("jepa_studio"))
        .map_err(|e| format!("copying {}: {e}", bundled_pkg.display()))?;
    let pyproject = format!(
        "[build-system]\nrequires = [\"setuptools>=69\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n\n\
         [project]\nname = \"jepa-studio\"\nversion = \"{version}\"\nrequires-python = \">=3.10\"\n\n\
         [tool.setuptools.packages.find]\ninclude = [\"jepa_studio*\"]\n\n\
         [tool.setuptools.package-data]\njepa_studio = [\"*.json\", \"**/*.json\"]\n"
    );
    fs::write(staging.join("pyproject.toml"), pyproject).map_err(|e| e.to_string())
}

fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "__pycache__" {
            continue;
        }
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_tree(&entry.path(), &dst.join(&name))?;
        } else if ty.is_file() {
            fs::copy(entry.path(), dst.join(&name))?;
        }
    }
    Ok(())
}

pub struct SetupPaths {
    pub data_dir: PathBuf,
    pub bundled_pkg: PathBuf,
}

/// Create or refresh the private environment. Returns the interpreter to run the backend with.
pub fn ensure_env(
    paths: &SetupPaths,
    app_version: &str,
    log: &Logger,
    progress: Progress,
) -> Result<SetupRecord, String> {
    let record_path = paths.data_dir.join("setup.json");
    let want = stamp(REQUIREMENTS, app_version);
    if let Some(rec) = read_record(&record_path) {
        if record_is_current(&rec, &want) {
            log.line("setup.json is current; skipping install");
            return Ok(rec);
        }
    }

    progress(Some(0.02), "Looking for Python 3.10+", "");
    let (base, ver) = find_python()?;
    log.line(&format!("base python: {} ({}.{})", base.display(), ver.0, ver.1));

    let venv = paths.data_dir.join("venv");
    let py = venv_python(&venv);
    if !py.is_file() {
        progress(Some(0.04), "Creating a private Python environment", &venv.display().to_string());
        let mut c = base.command();
        c.args(["-m", "venv", "--clear"]).arg(&venv);
        run_logged(c, log, &|_| {}).map_err(|e| {
            format!("Could not create the virtual environment (on Debian/Ubuntu install python3-venv).\n{e}")
        })?;
    }

    progress(Some(0.08), "Updating pip", "");
    let mut c = pip(&py);
    c.args(["install", "--upgrade", "pip"]);
    run_logged(c, log, &|l| progress(None, "Updating pip", l))?;

    progress(Some(0.1), "Detecting your accelerator", "");
    let det = detect_hardware();
    let mut choice = torch_choice(Os::current(), &det);
    if let Ok(forced) = std::env::var("JEPA_STUDIO_TORCH_INDEX") {
        if !forced.trim().is_empty() {
            choice.indexes = vec![Some(forced.trim().to_string())];
        }
    }
    log.line(&format!("hardware: {det:?} -> {choice:?}"));

    let pins = torch_pins(REQUIREMENTS);
    let mut tried = Vec::new();
    let mut used = None;
    let mut last_err = String::new();
    for (i, index) in choice.indexes.iter().enumerate() {
        let label = index.clone().unwrap_or_else(|| "pypi".into());
        tried.push(label.clone());
        let head = format!("Installing PyTorch ({}) from {label}", choice.accelerator);
        progress(Some(0.12), &head, "");
        let mut c = pip(&py);
        c.arg("install").args(&pins);
        if let Some(u) = index {
            c.args(["--index-url", u]);
        }
        let n = Mutex::new(0u32);
        let r = run_logged(c, log, &|l| {
            let mut k = n.lock().unwrap();
            if l.starts_with("Collecting") || l.starts_with("Downloading") || l.contains("Installing collected") {
                *k += 1;
            }
            // Asymptotic progress between 12% and 70%: the torch download dominates.
            let f = 0.12 + 0.58 * (1.0 - (-(*k as f64) / 12.0).exp());
            progress(Some(f), &head, l);
        });
        match r {
            Ok(()) => {
                used = Some(label);
                if i + 1 == choice.indexes.len() && choice.indexes.len() > 1 {
                    choice.accelerator = "cpu".into(); // every accelerated index failed
                }
                break;
            }
            Err(e) => {
                log.line(&format!("torch install from {label} failed, trying the next index"));
                last_err = e;
            }
        }
    }
    let torch_index = used
        .ok_or_else(|| format!("Could not install PyTorch from any wheel index ({}).\n{last_err}", tried.join(", ")))?;

    progress(Some(0.72), "Installing the remaining pinned packages", "");
    let req_file = paths.data_dir.join("requirements-desktop.txt");
    fs::write(&req_file, REQUIREMENTS).map_err(|e| e.to_string())?;
    let mut c = pip(&py);
    c.args(["install", "-r"]).arg(&req_file);
    run_logged(c, log, &|l| progress(None, "Installing the remaining pinned packages", l))?;

    progress(Some(0.9), "Installing jepa-studio", "");
    let staging = paths.data_dir.join("pkg-src");
    stage_package(&paths.bundled_pkg, &staging, app_version)?;
    let mut c = pip(&py);
    c.args(["install", "--no-deps", "--force-reinstall", "--no-cache-dir"]).arg(&staging);
    run_logged(c, log, &|l| progress(None, "Installing jepa-studio", l))?;
    let _ = fs::remove_dir_all(&staging);

    let rec = SetupRecord {
        schema: 1,
        app_version: app_version.into(),
        stamp: want,
        base_python: base.display(),
        python_version: format!("{}.{}", ver.0, ver.1),
        venv_python: py,
        accelerator: choice.accelerator,
        torch_index,
        torch_indexes_tried: tried,
        completed_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    fs::write(&record_path, serde_json::to_string_pretty(&rec).unwrap()).map_err(|e| e.to_string())?;
    Ok(rec)
}

/// No console window flashes on Windows for helper processes.
#[allow(unused_mut)]
pub fn quiet(mut c: Command) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv(driver: Option<(u32, u32)>, cc: Option<(u32, u32)>) -> Detected {
        Detected { nvidia: Some(NvidiaInfo { driver_cuda: driver, min_compute_cap: cc }), rocm: false }
    }
    fn tags(c: &TorchChoice) -> Vec<String> {
        c.indexes
            .iter()
            .map(|i| i.as_deref().map(|u| u.rsplit('/').next().unwrap().to_string()).unwrap_or("pypi".into()))
            .collect()
    }

    #[test]
    fn macos_uses_pypi_even_with_other_flags() {
        let c = torch_choice(Os::Mac, &nv(Some((13, 0)), None));
        assert_eq!(c.accelerator, "mps");
        assert_eq!(c.indexes, vec![None]);
    }

    #[test]
    fn nvidia_new_driver_new_gpu_prefers_newest_cuda() {
        let c = torch_choice(Os::Linux, &nv(Some((13, 0)), Some((8, 9))));
        assert_eq!(c.accelerator, "nvidia");
        assert_eq!(tags(&c), ["cu130", "cu128", "cu126", "cpu"]);
        assert_eq!(c.indexes[0].as_deref(), Some("https://download.pytorch.org/whl/cu130"));
    }

    #[test]
    fn nvidia_driver_limits_cuda_version() {
        let c = torch_choice(Os::Windows, &nv(Some((12, 6)), Some((8, 6))));
        assert_eq!(tags(&c), ["cu126", "cpu"]);
        let c = torch_choice(Os::Windows, &nv(Some((12, 9)), None));
        assert_eq!(tags(&c), ["cu128", "cu126", "cpu"]);
    }

    #[test]
    fn nvidia_old_gpu_skips_builds_without_its_kernels() {
        // Pascal (6.1): CUDA 13 and cu128 wheels dropped it.
        let c = torch_choice(Os::Linux, &nv(Some((13, 0)), Some((6, 1))));
        assert_eq!(tags(&c), ["cu126", "cpu"]);
    }

    #[test]
    fn nvidia_driver_too_old_falls_back_to_cpu() {
        let c = torch_choice(Os::Linux, &nv(Some((11, 8)), Some((8, 0))));
        assert_eq!(c.accelerator, "cpu");
        assert_eq!(tags(&c), ["cpu"]);
    }

    #[test]
    fn nvidia_unknown_details_tries_everything() {
        let c = torch_choice(Os::Linux, &nv(None, None));
        assert_eq!(tags(&c), ["cu130", "cu128", "cu126", "cpu"]);
    }

    #[test]
    fn rocm_only_on_linux() {
        let det = Detected { nvidia: None, rocm: true };
        let c = torch_choice(Os::Linux, &det);
        assert_eq!(c.accelerator, "rocm");
        assert_eq!(tags(&c), ["rocm7.0", "rocm6.4", "cpu"]);
        let c = torch_choice(Os::Windows, &det);
        assert_eq!((c.accelerator.as_str(), tags(&c)), ("cpu", vec!["cpu".to_string()]));
    }

    #[test]
    fn nvidia_wins_over_rocm_and_default_is_cpu() {
        let c = torch_choice(Os::Linux, &Detected { nvidia: Some(NvidiaInfo::default()), rocm: true });
        assert_eq!(c.accelerator, "nvidia");
        let c = torch_choice(Os::Linux, &Detected::default());
        assert_eq!((c.accelerator.as_str(), tags(&c)), ("cpu", vec!["cpu".to_string()]));
    }

    #[test]
    fn parses_nvidia_smi_output() {
        let banner = "| NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 13.0     |";
        assert_eq!(parse_driver_cuda(banner), Some((13, 0)));
        assert_eq!(parse_driver_cuda("no gpu"), None);
        assert_eq!(parse_min_compute_cap("8.6\n7.5\n"), Some((7, 5)));
        assert_eq!(parse_min_compute_cap("[N/A]\n"), None);
    }

    #[test]
    fn parses_python_version_and_pins() {
        assert_eq!(parse_python_version("3 12\n"), Some((3, 12)));
        assert_eq!(parse_python_version("garbage"), None);
        assert!(parse_python_version("3 9").unwrap() < MIN_PYTHON);
        let pins = torch_pins(REQUIREMENTS);
        assert_eq!(pins.len(), 2, "{pins:?}");
        assert!(pins.iter().any(|p| p.starts_with("torch==")));
        assert!(pins.iter().any(|p| p.starts_with("torchvision==")));
        assert_eq!(
            torch_pins("# torch==1\ntorch-geometric==2\nTorch == 2.1 ; python_version>'3'\n"),
            vec!["Torch == 2.1 ; python_version>'3'"]
        );
    }

    #[test]
    fn stamp_changes_with_inputs() {
        assert_eq!(stamp("a", "1").len(), 64);
        assert_ne!(stamp("a", "1"), stamp("a", "2"));
        assert_ne!(stamp("a", "1"), stamp("b", "1"));
    }
}
