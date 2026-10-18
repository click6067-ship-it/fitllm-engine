import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const workflowUrl = new URL('../.github/workflows/day0-watch.yml', import.meta.url);

// 2026-10-18: schedule 도 apply 잡을 돈다 — 발견이 읽기 전용 요약에만 남으면 아무도 행동하지 않았다.
// 바뀌지 않는 것: issues:write 는 apply 잡 하나, 증거 업로드가 쓰기보다 먼저, 수동 기본값은 dry-run.
test('schedule은 apply 잡을 돌고 수동 기본값은 read-only dry-run이며 issues write는 apply 잡 하나다', async () => {
  const workflow = await readFile(workflowUrl, 'utf8');
  assert.match(workflow, /schedule:\s*\n\s*- cron:/);
  assert.match(workflow, /default:\s*dry-run/);
  const dryRunJob = workflow.match(/\n  dry-run:[\s\S]*?(?=\n  apply-issues:)/)?.[0] || '';
  const applyJob = workflow.match(/\n  apply-issues:[\s\S]*$/)?.[0] || '';
  assert.match(dryRunJob, /if:\s*github\.event_name == 'workflow_dispatch' && inputs\.mode != 'apply-issues'/);
  assert.match(dryRunJob, /permissions:[\s\S]*?contents:\s*read[\s\S]*?issues:\s*read/);
  assert.match(applyJob, /if:\s*github\.event_name == 'schedule' \|\| \(github\.event_name == 'workflow_dispatch' && inputs\.mode == 'apply-issues'\)/);
  assert.match(applyJob, /permissions:[\s\S]*?contents:\s*read[\s\S]*?issues:\s*write/);
  assert.equal((workflow.match(/issues:\s*write/g) || []).length, 1);
  assert.doesNotMatch(workflow, /contents:\s*write|pull-requests:\s*write|deployments:\s*write|packages:\s*write/);
  // 쓰기는 업로드된 증거를 가리킨 뒤에만: plan → upload → apply 순서, apply 는 업로드 출력을 받는다.
  const plan = applyJob.indexOf('--plan --output-dir');
  const upload = applyJob.indexOf('actions/upload-artifact@');
  const apply = applyJob.indexOf('--apply-issues');
  assert.ok(plan > 0 && plan < upload && upload < apply, 'plan → upload → apply 순서가 깨졌다');
});

test('adapter는 day0 apply 잡 밖에서 issue를 쓰지 않는다', () => {
  const script = fileURLToPath(new URL('../.github/scripts/day0-watch.mjs', import.meta.url));
  const runApply = (env) => spawnSync(process.execPath, [script, '--apply-issues'], {
    encoding: 'utf8',
    env: { PATH: process.env.PATH, ...env },
  });
  for (const env of [
    { GITHUB_EVENT_NAME: 'push', DAY0_MODE: 'apply-issues' },
    { GITHUB_EVENT_NAME: 'pull_request', DAY0_MODE: 'apply-issues' },
    { GITHUB_EVENT_NAME: 'schedule' },
  ]) {
    const result = runApply(env);
    assert.equal(result.status, 1, JSON.stringify(env));
    assert.match(result.stderr, /issue apply requires the day0-watch apply-issues job/);
  }
  // schedule + apply 모드는 가드를 통과하고, 그다음 계약(plan·증거 URL·digest 필수)에서 멈춘다.
  const scheduled = runApply({ GITHUB_EVENT_NAME: 'schedule', DAY0_MODE: 'apply-issues' });
  assert.equal(scheduled.status, 1);
  assert.match(scheduled.stderr, /apply requires plan and uploaded artifact URL\/digest/);
});

test('workflow는 직렬화되고 apply evidence만 immutable 이름으로 업로드한다', async () => {
  const workflow = await readFile(workflowUrl, 'utf8');
  assert.match(workflow, /group:\s*day0-watch-\$\{\{ github\.repository \}\}/);
  assert.match(workflow, /cancel-in-progress:\s*false/);
  assert.equal((workflow.match(/actions\/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09/g) || []).length, 2);
  assert.equal((workflow.match(/actions\/setup-node@a0853c24544627f65ddf259abe73b1d18a591444/g) || []).length, 2);
  assert.match(workflow, /actions\/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f/);
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
