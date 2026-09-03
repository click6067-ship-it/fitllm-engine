import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const RELEASE = Object.freeze({
  repository: 'AlexsJones/llmfit',
  releaseTag: 'v1.1.12',
  binaryVersion: 'llmfit 1.1.12',
  artifactUrl: 'https://github.com/AlexsJones/llmfit/releases/download/v1.1.12/llmfit-v1.1.12-x86_64-unknown-linux-gnu.tar.gz',
  artifactSha256: '6a97338862c87e497c844ccd29a16512a147335631c179744b4f6cc87a36ead1',
});

const CASES = Object.freeze([
  { id: 'llama31-8b-gqa', role: 'control', architecture: 'gqa', modelInput: 'mlx-community/Llama-3.1-8B-Instruct-4bit', fitllmModel: 'Llama-3.1-8B-Instruct', stdoutFile: 'llama31-8b-gqa.json' },
  { id: 'gemma4-31b-sliding-window', role: 'counterexample', architecture: 'sliding-window', modelInput: 'google/gemma-4-31B-it', fitllmModel: 'Gemma 4 31b', stdoutFile: 'gemma4-31b-sliding-window.json' },
  { id: 'glm47-flash-mla', role: 'counterexample', architecture: 'mla', modelInput: 'zai-org/GLM-4.7-Flash', fitllmModel: 'GLM-4.7-Flash', stdoutFile: 'glm47-flash-mla.json' },
]);

const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const flag = (name) => {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
};

export function captureLlmfit({ binaryPath, artifactPath, outputDir }) {
  if (!binaryPath || !artifactPath) throw new Error('usage: node scripts/capture-llmfit.mjs --binary <path> --artifact <tar.gz>');
  const artifactHash = sha256(readFileSync(artifactPath));
  if (artifactHash !== RELEASE.artifactSha256) {
    throw new Error(`artifact SHA-256 mismatch: expected ${RELEASE.artifactSha256}, got ${artifactHash}`);
  }
  const binaryVersion = execFileSync(binaryPath, ['--version'], { encoding: 'utf8', timeout: 10_000 }).trim();
  if (binaryVersion !== RELEASE.binaryVersion) throw new Error(`binary version mismatch: ${binaryVersion}`);

  mkdirSync(outputDir, { recursive: true });
  const captured = CASES.map((entry) => {
    const args = ['--json', 'plan', entry.modelInput, '--context', '8192', '--quant', 'Q4_K_M'];
    const stdout = execFileSync(binaryPath, args, { encoding: 'utf8', timeout: 30_000, maxBuffer: 4 * 1024 * 1024 });
    JSON.parse(stdout);
    writeFileSync(`${outputDir}/${entry.stdoutFile}`, stdout);
    return { ...entry, args, stdoutSha256: sha256(stdout) };
  });
  const manifest = {
    schemaVersion: 1,
    evidenceClass: 'architecture_differential_not_runtime_accuracy',
    competitor: { ...RELEASE },
    cases: captured,
  };
  writeFileSync(`${outputDir}/manifest.json`, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const outputDir = dirname(fileURLToPath(new URL('../benchmarks/llmfit-v1.1.12/manifest.json', import.meta.url)));
    const manifest = captureLlmfit({ binaryPath: flag('--binary'), artifactPath: flag('--artifact'), outputDir });
    console.log(`captured ${manifest.cases.length} llmfit cases with verified ${manifest.competitor.artifactSha256}`);
  } catch (error) {
    console.error(error.message);
    process.exit(2);
  }
}
