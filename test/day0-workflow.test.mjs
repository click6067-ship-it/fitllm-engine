import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';

const workflowUrl = new URL('../.github/workflows/day0-watch.yml', import.meta.url);

test('schedule/default dispatch는 read-only dry-run이고 apply만 issues write다', async () => {
  const workflow = await readFile(workflowUrl, 'utf8');
  assert.match(workflow, /default:\s*dry-run/);
  assert.match(workflow, /dry-run:[\s\S]*?permissions:[\s\S]*?contents:\s*read[\s\S]*?issues:\s*read/);
  assert.match(workflow, /apply-issues:[\s\S]*?if:\s*github\.event_name == 'workflow_dispatch' && inputs\.mode == 'apply-issues'/);
  assert.match(workflow, /apply-issues:[\s\S]*?permissions:[\s\S]*?contents:\s*read[\s\S]*?issues:\s*write/);
  assert.equal((workflow.match(/issues:\s*write/g) || []).length, 1);
  assert.doesNotMatch(workflow, /contents:\s*write|pull-requests:\s*write|deployments:\s*write|packages:\s*write/);
});

test('workflow는 직렬화되고 apply evidence만 immutable 이름으로 업로드한다', async () => {
  const workflow = await readFile(workflowUrl, 'utf8');
  assert.match(workflow, /group:\s*day0-watch-\$\{\{ github\.repository \}\}/);
  assert.match(workflow, /cancel-in-progress:\s*false/);
  assert.equal((workflow.match(/actions\/checkout@11bd71901bbe5b1630ceea73d27597364c9af683/g) || []).length, 2);
  assert.equal((workflow.match(/actions\/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020/g) || []).length, 2);
  assert.match(workflow, /actions\/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02/);
  assert.equal((workflow.match(/persist-credentials:\s*false/g) || []).length, 2);
  assert.doesNotMatch(workflow, /actions\/(?:checkout|setup-node|upload-artifact)@v\d/);
  assert.match(workflow, /fitllm-day0-\$\{\{ github\.run_id \}\}-\$\{\{ github\.run_attempt \}\}/);
  assert.match(workflow, /retention-days:\s*30/);
  assert.match(workflow, /overwrite:\s*false/);
  assert.match(workflow, /steps\.evidence\.outputs\.artifact-digest/);
  assert.match(workflow, /steps\.evidence\.outputs\.artifact-url/);
});

test('workflow에는 source/catalog/PR/publish/deploy mutation 명령이 없다', async () => {
  const workflow = await readFile(workflowUrl, 'utf8');
  assert.doesNotMatch(workflow, /npm\s+publish|git\s+push|gh\s+pr|vercel|deploy/i);
  const dryRunJob = workflow.match(/\n  dry-run:[\s\S]*?(?=\n  apply-issues:)/)?.[0] || '';
  assert.doesNotMatch(dryRunJob, /upload-artifact|--apply-issues/);
});
