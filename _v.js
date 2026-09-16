const fs = require('fs');
const path = 'D:/Github Longterm Storage/priceprediction-noncontainer/price_predictionv3/setup_cuml_cuda.py';
const c = fs.readFileSync(path, 'utf8');

let depth_paren = 0, depth_bracket = 0, depth_brace = 0;
let in_string = null;
let in_comment = false;
for (let i = 0; i < c.length; i++) {
  const ch = c[i];
  const next = c[i+1] || '';
  if (in_comment) {
    if (ch === '\n') in_comment = false;
    continue;
  }
  if (in_string) {
    if (ch === '\\') { i++; continue; }
    if (in_string === '"""' && ch === '"' && c[i+1] === '"' && c[i+2] === '"') { in_string = null; i+=2; continue; }
    if (in_string === "'''" && ch === "'" && c[i+1] === "'" && c[i+2] === "'") { in_string = null; i+=2; continue; }
    if (ch === in_string || (in_string.length === 1 && ch === in_string)) { in_string = null; continue; }
    continue;
  }
  if (ch === '#') { in_comment = true; continue; }
  if (ch === '"' && next === '"' && c[i+2] === '"') { in_string = '"""'; i+=2; continue; }
  if (ch === "'" && next === "'" && c[i+2] === "'") { in_string = "'''"; i+=2; continue; }
  if (ch === '"') { in_string = '"'; continue; }
  if (ch === "'") { in_string = "'"; continue; }
  if (ch === '(') depth_paren++;
  if (ch === ')') depth_paren--;
  if (ch === '[') depth_bracket++;
  if (ch === ']') depth_bracket--;
  if (ch === '{') depth_brace++;
  if (ch === '}') depth_brace--;
}
console.log(`Final depths: paren=${depth_paren}, bracket=${depth_bracket}, brace=${depth_brace}`);

// Verify the fix
console.log(`\nVerification:`);
console.log(`  nvidia-cccl-cu12 removed: ${!c.includes('nvidia-cccl-cu12')}`);
console.log(`  nvidia-ptxcompiler-cu12 removed: ${!c.includes('nvidia-ptxcompiler-cu12')}`);
console.log(`  nvidia-cuda-runtime-cu12 in CONDA only: ${c.includes('nvidia-cuda-runtime-cu12')}`);
console.log(`  cuml-cu12 in PIP_PACKAGES: ${c.includes('cuml-cu12=={CUML_VERSION}')}`);
console.log(`  cupy-cuda12x in PIP_PACKAGES: ${c.includes('cupy-cuda12x')}`);
console.log(`  --extra-index-url removed: ${!c.includes('--extra-index-url')}`);
console.log(`  cuda-toolkit mentioned in comment: ${c.includes('cuda-toolkit')}`);
