import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const ROOT = new URL('..', import.meta.url).pathname;
const ACTION = new URL('../action.yml', import.meta.url);
const ENTRY = new URL('../scripts/action-entry.sh', import.meta.url).pathname;

function runAction(inputs) {
  const outputFile = join(mkdtempSync(join(tmpdir(), 'fitllm-action-')), 'output');
  const result = spawnSync('bash', [ENTRY], {
    cwd: ROOT,
    encoding: 'utf8',
    env: {
      ...process.env,
      GITHUB_ACTION_PATH: ROOT,
      GITHUB_OUTPUT: outputFile,
      INPUT_MODEL: '', INPUT_GPU: '', INPUT_MAC: '', INPUT_QUANT: '', INPUT_CTX: '8192', INPUT_KV: '16', INPUT_COUNT: '1',
      ...inputs,
    },
  });
  return { ...result, output: readFileSync(outputFile, 'utf8') };
}

test('composite metadata passes expressions through env, never the shell command', () => {
  const source = readFileSync(ACTION, 'utf8');
  assert.match(source, /using:\s*['"]?composite/);
  assert.match(source, /INPUT_MODEL:\s*\$\{\{ inputs\.model \}\}/);
  assert.match(source, /INPUT_GPU:\s*\$\{\{ inputs\.gpu \}\}/);
  const runBlocks = [...source.matchAll(/run:\s*([^\n]*(?:\n\s{8,}[^\n]*)*)/g)].map((match) => match[1]).join('\n');
  assert.doesNotMatch(runBlocks, /\$\{\{\s*inputs\./);
  assert.match(source, /value:\s*\$\{\{ steps\.preflight\.outputs\.result \}\}/);
});

test('entrypoint constructs an argv array and validates exactly one device', () => {
  const source = readFileSync(ENTRY, 'utf8');
  assert.match(source, /args=\("\$INPUT_MODEL"\)/);
  assert.match(source, /args\+=\(--gpu "\$INPUT_GPU"\)/);
  assert.match(source, /"\$\{args\[@\]\}"/);
  assert.doesNotMatch(source, /eval|bash\s+-c/);

  const neither = runAction({ INPUT_MODEL: 'Llama-3.1-8B-Instruct' });
  assert.equal(neither.status, 2);
  assert.match(neither.stderr, /exactly one of gpu or mac/i);
  const both = runAction({ INPUT_MODEL: 'Llama-3.1-8B-Instruct', INPUT_GPU: 'RTX 4090', INPUT_MAC: '64' });
  assert.equal(both.status, 2);
});

test('composite action propagates CLI fit, no-fit, and invalid exit contracts', () => {
  const fits = runAction({ INPUT_MODEL: 'Llama-3.1-8B-Instruct', INPUT_GPU: 'RTX 4090' });
  assert.equal(fits.status, 0);
  assert.match(fits.output, /exit-code=0/);
  assert.match(fits.output, /"verdict": "yes"/);
  assert.match(fits.output, /"basis": \{/);

  const noFit = runAction({ INPUT_MODEL: 'GLM-5.2', INPUT_GPU: 'RTX 4090' });
  assert.equal(noFit.status, 1);
  assert.match(noFit.output, /exit-code=1/);
  assert.match(noFit.output, /"verdict": "no"/);

  const invalid = runAction({ INPUT_MODEL: 'not-a-model', INPUT_GPU: 'RTX 4090' });
  assert.equal(invalid.status, 2);
  assert.match(invalid.output, /exit-code=2/);
});

test('multiline output delimiter cannot be injected with a model input line', () => {
  const attempted = runAction({
    INPUT_MODEL: 'not-a-model\nFITLLM_RESULT_EOF\npwned=yes',
    INPUT_GPU: 'RTX 4090',
  });
  assert.equal(attempted.status, 2);
  const lines = attempted.output.trimEnd().split('\n');
  const delimiter = lines[0].replace('result<<', '');
  assert.notEqual(delimiter, 'FITLLM_RESULT_EOF');
  assert.equal(lines.filter((line) => line === delimiter).length, 1);
  assert.equal(lines.at(-1), 'exit-code=2');
});

test('action forwards conditional premises and preserves exit contracts', () => {
  const affected = runAction({ INPUT_MODEL: 'GLM-4.7-Flash', INPUT_GPU: 'RTX 4090' });
  assert.equal(affected.status, 0);
  assert.match(affected.output, /mla-compressed-latent-cache/);
  const plain = runAction({ INPUT_MODEL: 'Llama-3.1-8B-Instruct', INPUT_GPU: 'RTX 4090' });
  assert.equal(plain.status, 0);
  assert.doesNotMatch(plain.output, /structuralAssumptions/);
  const noFit = runAction({ INPUT_MODEL: 'GLM-5.2', INPUT_GPU: 'RTX 4090' });
  assert.equal(noFit.status, 1);
  const invalid = runAction({ INPUT_MODEL: 'not-a-model', INPUT_GPU: 'RTX 4090' });
  assert.equal(invalid.status, 2);
});
