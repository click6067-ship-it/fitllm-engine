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
  const testJob = job('test')
  const pack = job('pack')
  const publish = job('publish')
  const postcondition = job('postcondition')

  assert.match(workflow, /^permissions: \{\}$/m)
  assert.equal(workflow.match(/id-token:\s*write/g)?.length, 1)
  assert.doesNotMatch(verify, /id-token:\s*write/)
  assert.doesNotMatch(testJob, /id-token:\s*write/)
  assert.doesNotMatch(pack, /id-token:\s*write/)
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
  const pack = job('pack')
  const publish = job('publish')
  assert.match(pack, /release-manifest\.tsv/)
  assert.match(pack, /sha512-/)
  assert.match(pack, /actions\/upload-artifact@[0-9a-f]{40}/)
  assert.match(pack, /retention-days: 1/)
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
  const pack = job('pack')
  const publish = job('publish')
  const postcondition = job('postcondition')
  assert.match(pack, /Payload parity/)
  assert.match(pack, /package name/)
  assert.match(pack, /npm pack --ignore-scripts/)
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

test('test_publish_config_blocked_and_registry_forced: tarball config cannot redirect publishing', () => {
  const publish = job('publish')
  assert.match(publish, /inspect_tarball_manifest\(\)/)
  assert.equal(publish.match(/^\s+inspect_tarball_manifest "release-artifacts\//gm)?.length, 2)
  assert.match(publish, /tar -xOf/)
  assert.match(publish, /Object\.hasOwn\(manifest, 'publishConfig'\)/)
  assert.match(publish, /assert\.equal\(manifest\.name, expectedName\)/)
  assert.match(publish, /assert\.equal\(manifest\.version, expectedVersion\)/)

  const manifestGate = publish.indexOf('inspect_tarball_manifest "release-artifacts/')
  const firstPublish = publish.search(/^\s+npm publish /m)
  assert.ok(manifestGate > 0 && manifestGate < firstPublish)

  const publishCommands = publish.split('\n').filter((line) => /^\s+npm publish /.test(line))
  assert.equal(publishCommands.length, 2)
  for (const command of publishCommands) {
    assert.match(command, /npm publish "\.\/release-artifacts\/\$[A-Z]+_TARBALL"/)
    assert.match(command, /--registry=https:\/\/registry\.npmjs\.org\//)
    assert.match(command, /--strict-ssl=true/)
    assert.match(command, /--proxy=false/)
    assert.match(command, /--https-proxy=false/)
    assert.match(command, /--ignore-scripts/)
    assert.match(command, /--provenance/)
  }
})

test('test_pack_is_fresh_job: tested checkout is never reused to build release artifacts', () => {
  const verify = job('verify')
  const testJob = job('test')
  const pack = job('pack')
  const publish = job('publish')

  assert.doesNotMatch(verify, /npm (?:ci|test|pack)|upload-artifact/)
  assert.match(testJob, /^    needs: verify$/m)
  assert.match(testJob, /^          ref: \$\{\{ needs\.verify\.outputs\.source_sha \}\}$/m)
  assert.match(testJob, /npm test/)
  assert.match(testJob, /npm run census:check/)
  assert.doesNotMatch(testJob, /npm pack|upload-artifact/)

  assert.match(pack, /^    needs: \[verify, test\]$/m)
  assert.match(pack, /^          ref: \$\{\{ needs\.verify\.outputs\.source_sha \}\}$/m)
  assert.match(pack, /persist-credentials: false/)
  assert.match(pack, /\[ "\$HEAD_SHA" = "\$SOURCE_SHA" \]/)
  assert.match(pack, /git diff --exit-code/)
  assert.match(pack, /\[ -z "\$\(git status --porcelain\)" \]/)
  assert.doesNotMatch(pack, /git status --porcelain --untracked-files=no/)
  assert.match(pack, /npm pack --ignore-scripts/)
  assert.doesNotMatch(pack, /npm (?:ci|test)|npm run|import\('\.\/engine\.js'\)/)
  assert.match(pack, /actions\/upload-artifact@[0-9a-f]{40}/)
  assert.match(publish, /^    needs: \[verify, pack\]$/m)
})

test('test_master_history_and_release_authority: release SHA and authority are both gated', () => {
  const verify = job('verify')
  assert.match(verify, /refs\/remotes\/origin\/master/)
  assert.match(verify, /git merge-base --is-ancestor "\$PUSH_SHA" refs\/remotes\/origin\/master/)
  assert.match(verify, /approved master history/)
  assert.match(workflow, /readiness[\s\S]{0,300}(?:tag ruleset|protected release authority)[\s\S]{0,100}required human reviewer/i)
  assert.match(workflow, /^    environment: npm-publish$/m)
})
