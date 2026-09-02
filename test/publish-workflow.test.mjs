import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const workflow = readFileSync(new URL('../.github/workflows/publish.yml', import.meta.url), 'utf8')

function job(name) {
  const match = workflow.match(new RegExp(`^  ${name}:\\n[\\s\\S]*?(?=^  [a-zA-Z0-9_-]+:\\n|(?![\\s\\S]))`, 'm'))
  assert.ok(match, `missing ${name} job`)
  return match[0]
}

function runBodies(source = workflow) {
  const lines = source.split('\n')
  const bodies = []
  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(/^(\s*)run:\s*\|\s*$/)
    if (!match) continue
    const indent = match[1].length
    const body = []
    for (index += 1; index < lines.length; index += 1) {
      const line = lines[index]
      if (line.trim() && line.match(/^\s*/)[0].length <= indent) {
        index -= 1
        break
      }
      body.push(line)
    }
    bodies.push(body.join('\n'))
  }
  return bodies
}

test('test_dispatch_tag_env_exact: dispatch tag is env-only and anchored exactly', () => {
  assert.match(workflow, /^\s+DISPATCH_TAG: \$\{\{ inputs\.tag \}\}$/m)
  for (const body of runBodies()) assert.doesNotMatch(body, /\$\{\{\s*inputs\.tag\s*\}\}/)
  assert.match(workflow, /\[\[ "\$TAG" =~ \^v\[0-9\]\+\\\.\[0-9\]\+\\\.\[0-9\]\+\$ \]\]/)
  assert.match(workflow, /\[ "\$TRIGGER_REF_TYPE" = "tag" \]/)
  assert.match(workflow, /\[ "\$TAG" = "\$TRIGGER_REF_NAME" \]/)
})

test('test_oidc_minimal_job: OIDC exists only on the minimal protected publish job', () => {
  const verify = job('verify')
  const publish = job('publish')
  const postcondition = job('postcondition')

  assert.match(workflow, /^permissions: \{\}$/m)
  assert.equal(workflow.match(/id-token:\s*write/g)?.length, 1)
  assert.doesNotMatch(verify, /id-token:\s*write/)
  assert.doesNotMatch(postcondition, /id-token:\s*write/)
  assert.match(publish, /^    environment: npm-publish$/m)
  assert.match(publish, /^      id-token: write$/m)
  assert.doesNotMatch(publish, /actions\/checkout|npm (?:ci|test|pack)|node scripts\//)
  assert.match(publish, /^      NPM_CONFIG_IGNORE_SCRIPTS: 'true'$/m)

  for (const line of publish.split('\n').filter((candidate) => /npm publish/.test(candidate) && !/^\s*#/.test(candidate))) {
    assert.match(line, /--ignore-scripts/)
    assert.match(line, /\.tgz|TARBALL/)
  }
})

test('test_push_sha_and_exact_tools: release source and tools use immutable pins', () => {
  const verify = job('verify')
  assert.match(verify, /^      PUSH_SHA: \$\{\{ github\.sha \}\}$/m)
  assert.match(verify, /^          ref: \$\{\{ github\.sha \}\}$/m)
  assert.match(verify, /\[ "\$HEAD_SHA" = "\$PUSH_SHA" \]/)
  assert.match(verify, /\[ "\$TAG_SHA" = "\$PUSH_SHA" \]/)
  assert.match(workflow, /node-version: '22\.14\.0'/)
  assert.doesNotMatch(workflow, /node-version:\s*['"]?(?:22|24|lts|latest)['"]?\s*$/m)
  assert.ok((workflow.match(/npm install --global npm@11\.5\.1 --ignore-scripts/g)?.length ?? 0) >= 3)

  const actionUses = [...workflow.matchAll(/^\s+(?:- )?uses: (actions\/[^@\s]+)@([^\s#]+)/gm)]
  assert.ok(actionUses.length >= 4)
  for (const [, actionName, revision] of actionUses) {
    assert.match(revision, /^[0-9a-f]{40}$/, `${actionName} is not pinned to a full commit`)
  }
})

test('test_immutable_integrity_artifact: both registry versions are gated by the packed SHA512 values', () => {
  const verify = job('verify')
  const publish = job('publish')
  assert.match(verify, /release-manifest\.tsv/)
  assert.match(verify, /sha512-/)
  assert.match(verify, /actions\/upload-artifact@[0-9a-f]{40}/)
  assert.match(verify, /retention-days: 1/)
  assert.match(publish, /actions\/download-artifact@[0-9a-f]{40}/)
  assert.match(publish, /release-manifest\.tsv/)
  assert.match(publish, /EXPECTED_ENGINE_INTEGRITY/)
  assert.match(publish, /EXPECTED_ALIAS_INTEGRITY/)

  const firstPublish = publish.indexOf('npm publish')
  assert.ok(firstPublish > 0)
  const preflight = publish.slice(0, firstPublish)
  assert.match(preflight, /fitllm-engine/)
  assert.match(preflight, /fitllm/)
  assert.match(preflight, /dist\.integrity/)
  assert.ok((publish.match(/verify_registry_integrity/g)?.length ?? 0) >= 3)
  assert.ok((publish.match(/!= expected|!= \$expected|"\$actual" != "\$expected"/g)?.length ?? 0) >= 1)
  assert.ok((publish.match(/else\n\s+status=\$\?/g)?.length ?? 0) >= 2)
})

test('test_payload_parity_and_provenance: parity is checked before upload and provenance after publish', () => {
  const verify = job('verify')
  const publish = job('publish')
  const postcondition = job('postcondition')
  assert.match(verify, /Payload parity/)
  assert.match(verify, /package name/)
  assert.match(verify, /npm pack --ignore-scripts/)
  assert.doesNotMatch(publish, /Payload parity|npm pack/)
  assert.match(postcondition, /dist\.attestations\.provenance\.predicateType/)
  assert.match(postcondition, /https:\/\/slsa\.dev\/provenance\/v1/)
  assert.match(postcondition, /npm audit signatures/)
  assert.match(postcondition, /--ignore-scripts/)
  assert.match(postcondition, /EXPECTED_ENGINE_INTEGRITY/)
  assert.match(postcondition, /EXPECTED_ALIAS_INTEGRITY/)
})

test('test_readiness_external_gate: readiness remains false by default and gates OIDC publication', () => {
  const publish = job('publish')
  const executableLines = workflow.split('\n').filter((line) => !/^\s*#/.test(line)).join('\n')
  assert.match(publish, /READY: \$\{\{ vars\.NPM_TRUSTED_PUBLISHING_READY \}\}/)
  assert.match(publish, /if \[ "\$READY" != "true" \]/)
  assert.doesNotMatch(executableLines, /NPM_TRUSTED_PUBLISHING_READY\s*[:=]\s*true/)
  assert.match(workflow, /trusted publisher[\s\S]{0,200}environment[\s\S]{0,100}npm-publish/i)
  assert.match(workflow, /protect(?:ed)?[\s\S]{0,100}environment[\s\S]{0,100}selected-tag policy[\s\S]{0,50}v\*/i)
})
