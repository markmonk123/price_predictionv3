const fs = require('fs');
const path = 'D:/Github Longterm Storage/priceprediction-noncontainer/price_predictionv3/setup_cuml_cuda.py';
let c = fs.readFileSync(path, 'utf8');

// Fix 1: replace PIP_PACKAGES with a minimal list that pip can actually resolve.
// The old list included nvidia-cccl-cu12 and nvidia-ptxcompiler-cu12 which are
// NOT published to either PyPI or the NVIDIA index. Just install cuml-cu12 and
// cupy-cuda12x and let pip's resolver pull in cuda-toolkit[cublas,...] which
// brings the proper CUDA 12 runtime libs transitively.
const oldPkg = `# pip fallback packages (cuML 26.x on PyPI as cuml-cu12)
PIP_PACKAGES = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cccl-cu12",
    "nvidia-ptxcompiler-cu12",
    "nvidia-cublas-cu12",
    f"cuml-cu12=={CUML_VERSION}",
    "cupy-cuda12x",
]`;

const newPkg = `# pip fallback packages: cuML 26.x and cupy live on PyPI proper.
# pip's resolver will pull in cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]
# (CUDA 12 family) automatically as a transitive dependency of cuml-cu12.
# Do NOT pin nvidia-cccl-cu12 or nvidia-ptxcompiler-cu12 -- those names are
# not published to PyPI; the [cccl] / [nvptxcompiler] extras of cuda-toolkit
# are the canonical CUDA 12 source.
PIP_PACKAGES = [
    f"cuml-cu12=={CUML_VERSION}",
    "cupy-cuda12x>=13",
]`;

// Fix 2: drop the --extra-index-url flag from install_with_pip. The NVIDIA
// index only hosts cuml/cupy/cudf wheels; cuda-toolkit is on PyPI proper, so
// we don't need (and shouldn't add) the extra index.
const oldInstall = `def install_with_pip() -> bool:
    """Fallback: install cuML via the NVIDIA PyPI index."""
    info("Installing cuML via pip + NVIDIA index")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    cmd += PIP_PACKAGES
    cmd += ["--extra-index-url", "https://pypi.nvidia.com"]
    res = run(cmd)
    if res.returncode != 0:
        fail("pip install of cuML failed.")
        return False
    ok("cuML installed via pip")
    return True`;

const newInstall = `def install_with_pip() -> bool:
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
    return True`;

let changes = [];
if (c.includes(oldPkg)) {
  c = c.replace(oldPkg, newPkg);
  changes.push('PIP_PACKAGES replaced');
} else {
  changes.push('PIP_PACKAGES block NOT found (check encoding)');
}

if (c.includes(oldInstall)) {
  c = c.replace(oldInstall, newInstall);
  changes.push('install_with_pip updated');
} else {
  changes.push('install_with_pip block NOT found');
}

if (changes.every(x => x.includes('NOT'))) {
  console.log('FAILURES:');
  for (const x of changes) console.log('  ', x);
} else {
  fs.writeFileSync(path, c);
  console.log(changes.join('\n'));
}
