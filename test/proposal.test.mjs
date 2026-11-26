import assert from 'node:assert/strict';
import test from 'node:test';
import {AccessLedger, SourceReader} from '../src/access.mjs';
import {accumulateTrainingGradient, createTrainingItem} from '../src/training.mjs';
import {commitUpdate, paramsFromSlots, proposeUpdate, snapshotReads} from '../src/proposal.mjs';
import {frozenConstraints, mixtureDirection, portValue} from '../src/views.mjs';
import {sparseDot, sparseNorm} from '../src/linear.mjs';
import {syntheticAssets} from './fixture.mjs';

function readerFor(assets, policy) {
  const ledger = new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: assets.workBytes,
    dimension: assets.dimension,
    policy,
  });
  return new SourceReader(assets, {ledger, policy});
}

function learn(assets, policy, specs) {
  const reader = readerFor(assets, policy);
  const items = specs.map((spec) =>
    createTrainingItem(assets, {
      kind: spec.fields.length === 2 ? 'pair' : 'triple',
      fields: spec.fields,
      profile: spec.profile ?? 'rank4',
      exposure: spec.exposure ?? policy,
      lossWeight: spec.lossWeight ?? 1,
      upstream: spec.upstream ?? 1,
    }),
  );
  return accumulateTrainingGradient(assets, reader, items, {
    profile: specs[0]?.profile ?? 'rank4',
    exposure: policy,
  });
}

test('合法步是离原始梯度最近的欧氏投影，且所有有效冻结视图不变', () => {
  const assets = syntheticAssets();
  const training = learn(assets, 'open', [{fields: [0, 1]}, {fields: [5, 2]}, {fields: [3, 4]}]);
  const proposal = proposeUpdate(assets, {training, exposure: 'open', profile: 'rank4'});
  assert.equal(proposal.status, 'committed');

  const {constraints} = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  for (const check of proposal.constraints) {
    assert.ok(check.satisfied, `${check.name} dot=${check.dot}`);
  }
  // 投影后范数不超过原始步。
  assert.ok(proposal.stepNorm <= proposal.rawNorm + 1e-12);
  // 报告里每个有效冻结读口的变化都为零。
  for (const port of proposal.ports.filter((entry) => entry.effective)) {
    assert.ok(port.preserved, `${port.name} changed by ${port.change}`);
  }
  // 固定视图中 slot 20 的变化与 raw 不同，证明确实做了投影。
  const readOnly = constraints.find((constraint) => constraint.name === 'read-only');
  assert.ok(Math.abs(sparseDot(readOnly.direction, proposal.delta)) < 1e-10);
});

test('固定视图可部分为零：满足不等式梯度允许分量恰为 0', () => {
  const assets = syntheticAssets();
  const training = learn(assets, 'open', [{fields: [0, 1]}]);
  const proposal = proposeUpdate(assets, {training, exposure: 'open', profile: 'rank4'});
  assert.equal(proposal.status, 'committed');
  // 保留某些分量，但投影后至少有一个轴恰好为零。
  const slot20 = proposal.delta.get(20);
  assert.ok(slot20, 'slot 20 keeps an applicable component');
  assert.ok([...slot20].some((value) => value === 0));
  assert.ok([...slot20].some((value) => value !== 0));
});

test('变更公共向量后冻结读口仍保持原值；提交后端口差为零', () => {
  const assets = syntheticAssets();
  const keys = [20, 1, 2, 21, 3, 4];
  const base = paramsFromSlots(assets, keys);
  const before = {};
  const {constraints} = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  for (const constraint of constraints) {
    before[constraint.name] = portValue(constraint.direction, base);
  }

  const training = learn(assets, 'open', [{fields: [0, 1]}, {fields: [5, 2]}]);
  const proposal = proposeUpdate(assets, {training, exposure: 'open', profile: 'rank4'});
  const committed = commitUpdate(assets, base, proposal);
  assert.equal(committed.applied, true);

  for (const constraint of constraints) {
    const after = portValue(constraint.direction, committed.params);
    assert.ok(Math.abs(after - before[constraint.name]) < 1e-9, `${constraint.name}`);
  }
  // 但参数确实变了。
  let moved = 0;
  for (const [slot, vector] of committed.params) {
    const old = base.get(slot);
    for (let i = 0; i < vector.length; i += 1) if (Math.abs(vector[i] - old[i]) > 1e-9) moved += 1;
  }
  assert.ok(moved > 0);
});

test('冻结读口是几个许可域的联合视图，各域变化可以刚好抵消', () => {
  const assets = syntheticAssets();
  const mixture = assets.protectedMixtures.find((entry) => entry.name === 'joint');
  const {direction} = mixtureDirection(assets, mixture);

  // 域 5 与域 2 各自产生非零变化，但在 joint 内相互抵消。
  const training = learn(assets, 'open', [{fields: [5, 2]}, {fields: [0, 1]}]);
  const proposal = proposeUpdate(assets, {training, exposure: 'open', profile: 'rank4'});
  const slot20 = proposal.delta.get(20);
  const slot2 = proposal.delta.get(2);
  assert.ok(sparseNorm(new Map([[20, slot20 ?? new Float64Array(assets.dimension)]])) > 0);
  assert.ok(sparseNorm(new Map([[2, slot2 ?? new Float64Array(assets.dimension)]])) > 0);
  assert.ok(Math.abs(sparseDot(direction, proposal.delta)) < 1e-9);
});

test('逐域清零与只看公有槽都不等于合法投影', () => {
  const assets = syntheticAssets();
  const training = learn(assets, 'open', [{fields: [0, 1]}, {fields: [5, 2]}]);
  const proposal = proposeUpdate(assets, {training, exposure: 'open', profile: 'rank4'});
  assert.equal(proposal.status, 'committed');

  // 逐域清零：把域 0 的私有槽和共享槽全部抹掉。
  const naive = new Map();
  for (const [slot, vector] of training.gradient) naive.set(slot, Float64Array.from(vector));
  const afterPerField = new Map(naive);
  afterPerField.delete(20);
  afterPerField.delete(1);
  assert.notEqual(sparseNorm(afterPerField), 0);
  // 合法投影保留了 slot 20 上的一部分方向。
  assert.ok(sparseNorm(proposal.delta) > 0);
  assert.notDeepEqual([...(proposal.delta.get(20) ?? [])], [...(afterPerField.get(20) ?? [])]);
});

test('被阻止的提案不会写入任何分量，并保留上一次合法读法', () => {
  const assets = syntheticAssets();
  const training = learn(assets, 'open', [{fields: [0, 1]}]);
  const snapshot = {
    policy: 'open',
    profile: 'rank4',
    ports: [{name: 'read-only', kind: 'view', value: 1.5, frozen: true}],
  };
  const proposal = proposeUpdate(assets, {
    training,
    exposure: 'tight',
    profile: 'rank4',
    lastLegalRead: snapshot,
  });
  assert.equal(proposal.status, 'blocked');
  assert.equal(proposal.blockReason, 'domain-switch');
  assert.deepEqual(proposal.fallback.ports, snapshot.ports);
  assert.ok(proposal.fallback.keep.includes('read-only'));

  const committed = commitUpdate(assets, paramsFromSlots(assets, [20, 1]), proposal);
  assert.equal(committed.applied, false);
  assert.equal(committed.params, null);
  assert.deepEqual(committed.fallback.ports, snapshot.ports);
  // 被阻止时仍报告可应用的向量分量，供调用方核对。
  assert.ok(proposal.applicable.delta instanceof Map);
});

test('允许部分提交时写入可用分量并保留被阻止关系', () => {
  const assets = syntheticAssets();
  const training = learn(assets, 'open', [{fields: [5, 2]}]);
  // 让提交身份变成 tight，但域 5 与 2 仍许可。
  const proposal = proposeUpdate(assets, {
    training,
    exposure: 'tight',
    profile: 'rank4',
    allowPartial: true,
  });
  // 来源身份是 open，因此 stale，但没有可写来源。
  assert.equal(proposal.status, 'blocked');
  assert.equal(proposal.blockedRelations.filter((entry) => entry.relation === 'gradient-source').length, 1);

  // 手工提供一份 tight 身份的梯度：部分提交应成功。
  const tightTraining = learn(assets, 'tight', [{fields: [5, 2]}]);
  const allowed = proposeUpdate(assets, {
    training: tightTraining,
    exposure: 'tight',
    profile: 'rank4',
    allowPartial: true,
  });
  assert.notEqual(allowed.status, 'blocked');
  assert.ok(allowed.written instanceof Map);
});

test('提交保留的读口按当时档位身份验算', () => {
  const assets = syntheticAssets();
  // 低档学习后以高档提交：新增轴上的损失不得复用。
  const training = learn(assets, 'open', [{fields: [3, 4], profile: 'rank6'}]);
  const proposal = proposeUpdate(assets, {training, exposure: 'open', profile: 'rank4'});
  assert.equal(proposal.status, 'committed');
  assert.ok(proposal.admissions.restricted.length > 0);
  assert.deepEqual(proposal.admissions.restricted[0].droppedAxes, [5]);
  // 提交向量在轴 5 上必须为零。
  for (const vector of proposal.delta.values()) assert.equal(vector[5], 0);
});

test('快照可以被直接用作 lastLegalRead', () => {
  const assets = syntheticAssets();
  const reader = readerFor(assets, 'tight');
  const snapshot = snapshotReads({
    policy: 'tight',
    profile: 'rank4',
    ports: [{name: 'read-only', kind: 'view', value: 0.25, frozen: true}],
    readouts: [],
  });
  assert.equal(snapshot.policy, 'tight');
  const proposal = proposeUpdate(assets, {
    gradient: new Map([[20, Float64Array.from([1, 0, 0, 0, 0, 0])]]),
    exposure: 'tight',
    profile: 'rank4',
    lastLegalRead: snapshot,
  });
  // slot 20 只有关闭域 0 与许可域 5 共享，别名仍可写，因此不应因别名被拒绝。
  assert.ok(!proposal.blockedRelations.some((entry) => entry.relation === 'shared-slot'));
  void reader;
});
