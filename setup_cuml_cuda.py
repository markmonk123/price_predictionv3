#!/usr/bin/env python3
"""
setup_cuml_cuda.py — Configure cuML/CUDA acceleration for enhanced_prediction.py

What this script does
---------------------
enhanced_prediction.py auto-detects cuML/CUDA at import time. It only works when:

  1. The OS is Linux (cuML has no native Windows wheels).
  2. An NVIDIA GPU is visible (driver >= 525 recommended).
  3. The `cuml` and `cupy` Python packages are installed.
  4. CUDA toolkit libraries (libcudart, etc.) are discoverable.

This script automates that setup:

  * On Windows: prints WSL2 + CUDA driver setup instructions, then attempts to
    run the rest of the steps inside WSL automatically.
  * On WSL / native Linux with an NVIDIA GPU: installs RAPIDS cuML via conda
    (preferred) or pip (fallback), then runs a verification pass.
  * On any system without an NVIDIA GPU: falls back to the CPU scikit-learn
    path (no-op install; just confirms dependencies).

After running, invoke enhanced_prediction.py from the same Python environment
that this script set up and CUDA_AVAILABLE will be True.

Usage
-----
    python setup_cuml_cuda.py                # auto-detect, install, verify
    python setup_cuml_cuda.py --verify-only  # just run verification, no install
    python setup_cuml_cuda.py --method conda # force conda install path
    python setup_cuml_cuda.py --method pip   # force pip install path
    python setup_cuml_cuda.py --print-wsl    # only print WSL instructions
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
CUML_VERSION = "26.4.0"            # must match requirements.txt
CUDA_MAJOR = "12"                  # CUDA 12.x is the RAPIDS 26.x baseline
PYTHON_MIN = (3, 10)               # RAPIDS 26.x supports 3.10+
PYTHON_MAX = (3, 13)               # and through 3.12/3.13
NVIDIA_DRIVER_MIN = 525            # CUDA 12.0 baseline

# RAPIDS conda install channels / packages
CONDA_PACKAGES = [
    f"cudf=={CUML_VERSION}",
    f"cuml=={CUML_VERSION}",
    f"cupy-cuda{CUDA_MAJOR}x",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cccl-cu12",
    "nvidia-ptxcompiler-cu12",
]

# pip fallback packages: cuML 26.x and cupy live on PyPI proper.
# pip's resolver will pull in cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]
# (CUDA 12 family) automatically as a transitive dependency of cuml-cu12.
# Do NOT pin nvidia-cccl-cu12 or nvidia-ptxcompiler-cu12 -- those names are
# not published to PyPI; the [cccl] / [nvptxcompiler] extras of cuda-toolkit
# are the canonical CUDA 12 source.
PIP_PACKAGES = [
    f"cuml-cu12=={CUML_VERSION}",
    "cupy-cuda12x>=13",
]


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _c(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def info(msg: str) -> None:
    print(_c("36", f"[INFO] {msg}"))


def ok(msg: str) -> None:
    print(_c("32", f"[ OK ] {msg}"))


def warn(msg: str) -> None:
    print(_c("33", f"[WARN] {msg}"))


def fail(msg: str) -> None:
    print(_c("31;1", f"[FAIL] {msg}"))


def header(msg: str) -> None:
    print()
    print(_c("1;34", f"=== {msg} ==="))


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_platform() -> dict:
    """Return a dict describing the runtime environment."""
    info = {
        "system": platform.system(),                 # Windows / Linux / Darwin
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version_info,
        "is_wsl": False,
        "has_nvidia_smi": False,
        "gpu_count": 0,
        "driver_version": None,
        "cuda_runtime": None,
    }

    # Detect WSL (Microsoft's /proc/version override is the most reliable signal).
    try:
        proc_version = Path("/proc/version").read_text().lower()
        info["is_wsl"] = "microsoft" in proc_version or "wsl" in proc_version
    except OSError:
        pass

    # Detect NVIDIA GPU via nvidia-smi on PATH.
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        info["has_nvidia_smi"] = True
        try:
            out = subprocess.check_output(
                [nvidia_smi, "--query-gpu=count,driver_version", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode().strip()
            if out:
                first_line = out.splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 2:
                    info["gpu_count"] = int(parts[0])
                    info["driver_version"] = parts[1]
        except Exception:
            pass

    # Check CUDA runtime via nvcc if present.
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.DEVNULL, timeout=10).decode()
            for line in out.splitlines():
                if "release" in line.lower():
                    info["cuda_runtime"] = line.strip().split()[-1]
                    break
        except Exception:
            pass

    return info


def print_wsl_instructions() -> None:
    """Print the manual WSL2 + CUDA setup steps for Windows users."""
    header("Windows host detected — WSL2 + CUDA setup instructions")
    print("""
enhanced_prediction.py auto-detects cuML/CUDA at import time. cuML only ships
Linux wheels, so the script must run inside WSL2 with NVIDIA GPU passthrough.

Step 1: Enable WSL2 and install Ubuntu
  wsl --install                  # default distro (Ubuntu)
  # then restart Windows if prompted

Step 2: Install the NVIDIA CUDA driver for WSL
  Download and run the standard Windows NVIDIA driver (>= 525):
    https://www.nvidia.com/Download/index.aspx
  The same driver exposes the GPU to both Windows AND WSL2. Do NOT install
  a separate CUDA toolkit inside WSL — the driver is enough for cuML.

Step 3: Verify the GPU is visible inside WSL
  wsl
  nvidia-smi                     # should list your GPU + driver version

Step 4: Clone this repo inside WSL (or access it from /mnt/c/...)
  cd /mnt/c/Users/<you>/Github\ Longterm\ Storage/priceprediction-noncontainer/price_predictionv3
  python3 setup_cuml_cuda.py

Step 5: Run enhanced_prediction.py from inside WSL
  python3 enhanced_prediction.py

The script will then handle cuML installation automatically.
""")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, streaming stdout/stderr."""
    info("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False, **kw)


def ensure_python_version(env: dict) -> bool:
    py = env["python"]
    if py < PYTHON_MIN or py > PYTHON_MAX:
        fail(
            f"Python {py.major}.{py.minor} not in supported range "
            f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]} .. {PYTHON_MAX[0]}.{PYTHON_MAX[1]}. "
            "cuML 26.x requires 3.10-3.12."
        )
        return False
    ok(f"Python {py.major}.{py.minor}.{py.micro}")
    return True


def ensure_gpu_available(env: dict) -> bool:
    if not env["has_nvidia_smi"]:
        warn("nvidia-smi not found on PATH.")
        warn("CUDA acceleration will be unavailable. The CPU scikit-learn path will be used.")
        return False
    if env["gpu_count"] <= 0:
        warn("nvidia-smi is installed but reports zero GPUs.")
        return False
    driver = env["driver_version"]
    if driver:
        try:
            major = int(driver.split(".")[0])
            if major < NVIDIA_DRIVER_MIN:
                warn(f"NVIDIA driver {driver} is older than {NVIDIA_DRIVER_MIN}. "
                     "cuML may fail to load or run slowly.")
            else:
                ok(f"NVIDIA driver {driver} ({env['gpu_count']} GPU)")
        except Exception:
            ok(f"NVIDIA driver {driver} ({env['gpu_count']} GPU)")
    return True


def install_with_conda() -> bool:
    """Install RAPIDS via micromamba/conda. Preferred path."""
    # Prefer micromamba (lightweight), fall back to conda.
    for exe in ("micromamba", "mamba", "conda"):
        if shutil.which(exe):
            break
    else:
        warn("No conda/mamba/micromamba found on PATH.")
        return False

    env_name = "rapids"
    info(f"Creating conda env '{env_name}' with RAPIDS {CUML_VERSION}")
    create = run([
        exe, "create", "-y", "-n", env_name, "-c", "rapidsai", "-c", "conda-forge",
        f"python=3.12",
        f"cuml=={CUML_VERSION}",
        "cupy",
        "scikit-learn",
        "pandas",
        "numpy",
    ])
    if create.returncode != 0:
        fail(f"{exe} create returned {create.returncode}")
        return False

    ok(f"Conda env '{env_name}' created. Activate with: {exe} activate {env_name}")
    return True


def install_with_pip() -> bool:
    """Install cuML via pip. cuml-cu12's transitive deps include the CUDA 12
    runtime via cuda-toolkit[cublas,cufft,curand,cusolver,cusparse,cccl,cudart]==12.*
    so pip's resolver handles all the nvidia-* libs for us."""
    info(f"Installing cuml-cu12=={CUML_VERSION} and cupy-cuda12x via pip")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    cmd += PIP_PACKAGES
    res = run(cmd)
    if res.returncode != 0:
        fail("pip install of cuML failed. Try --method conda for a more reliable path.")
        return False
    ok("cuML installed via pip")
    return True


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_installation() -> int:
    """Import cuML/cupy in a subprocess and report status. Returns 0 on success."""
    header("Verification: probing cuML/CUDA imports")
    code = """
import sys
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    print(f"CUDA_AVAILABLE=True  GPUs={n}")
    sys.exit(0)
except Exception as e:
    print(f"cupy probe failed: {e!r}")
try:
    import cuml
    print("cuml importable but cupy/GPU init failed; CPU fallback will be used.")
    sys.exit(2)
except Exception as e:
    print(f"cuml not importable: {e!r}")
    sys.exit(3)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    for line in out.splitlines():
        print("   " + line)
    if res.returncode == 0:
        ok("enhanced_prediction.py will pick CUDA_AVAILABLE=True at import.")
        return 0
    if res.returncode == 2:
        warn("cuML is installed but GPU init failed. Check driver / CUDA libs.")
        return 2
    fail("cuML not installed. Run without --verify-only to install.")
    if err:
        for line in err.splitlines()[-5:]:
            print("   " + line)
    return res.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true", help="Skip install; only verify.")
    parser.add_argument("--method", choices=("auto", "conda", "pip"), default="auto",
                        help="Install method (default: auto)")
    parser.add_argument("--print-wsl", action="store_true", help="Only print WSL instructions and exit.")
    args = parser.parse_args()

    header("Setup: cuML/CUDA for enhanced_prediction.py")
    env = detect_platform()
    info(f"Platform: {env['system']} {env['release']} ({env['machine']})")
    info(f"WSL: {env['is_wsl']}")
    info(f"Python: {env['python'].major}.{env['python'].minor}.{env['python'].micro}")

    # Windows host: print WSL instructions and exit.
    if env["system"] == "Windows" and not env["is_wsl"]:
        print_wsl_instructions()
        return 0
    if args.print_wsl:
        print_wsl_instructions()
        return 0

    # From here we are on Linux (native or WSL).
    if not ensure_python_version(env):
        return 1
    has_gpu = ensure_gpu_available(env)

    if args.verify_only:
        return verify_installation()

    if not has_gpu:
        warn("No NVIDIA GPU detected — skipping cuML install.")
        warn("enhanced_prediction.py will use the CPU scikit-learn path. "
             "All estimators still work; GPU acceleration will be disabled.")
        return 0

    method = args.method
    if method == "auto":
        method = "conda" if shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda") else "pip"

    success = False
    if method == "conda":
        success = install_with_conda()
        if not success:
            warn("Conda install failed; falling back to pip.")
            success = install_with_pip()
    else:
        success = install_with_pip()
        if not success:
            warn("pip install failed; falling back to conda.")
            success = install_with_conda()

    if not success:
        fail("cuML install failed via all available methods.")
        return 1

    # After install, sync pip-installed deps to make sure scikit-learn, etc.
    # are also present in the active environment.
    if REQUIREMENTS_FILE.exists():
        info("Syncing remaining pip dependencies from requirements.txt")
        sync = run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
        if sync.returncode != 0:
            warn("pip sync from requirements.txt reported non-zero exit; continuing.")

    return verify_installation()


if __name__ == "__main__":
    raise SystemExit(main())
