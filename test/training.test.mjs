import assert from 'node:assert/strict';
import test from 'node:test';
import {AccessLedger, SourceReader} from '../src/access.mjs';
import {accumulateTrainingGradient, admissibleContributions, createTrainingItem} from '../src/training.mjs';
import {termGradient, accumulateGradient} from '../src/interactions.mjs';
import {syntheticAssets} from './fixture.mjs';

function readerFor(assets, policy) {
  const ledger = new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: assets.workBytes,
    dimension: assets.dimension,
    policy,
  });
  return {ledger, reader: new SourceReader(assets, {ledger, policy})};
}

function item(assets, fields, options = {}) {
  return createTrainingItem(assets, {
    kind: fields.length === 2 ? 'pair' : 'triple',
    fields,
    profile: 'rank4',
    exposure: 'open',
    ...options,
  });
}

test('只有已发布、有损失权且别名可编辑的项回传', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');

  const unpublished = accumulateTrainingGradient(assets, reader, [item(assets, [0, 1], {published: false})], {
    profile: 'rank6',
    exposure: 'open',
  });
  assert.equal(unpublished.admitted, 0);
  assert.equal(unpublished.blocked[0].reason, 'unpublished');

  const unweighted = accumulateTrainingGradient(assets, reader, [item(assets, [0, 1], {lossWeight: 0})], {
    profile: 'rank6',
    exposure: 'open',
  });
  assert.equal(unweighted.admitted, 0);
  assert.equal(unweighted.blocked[0].reason, 'no-loss-weight');

  const open = readerFor(assets, 'open');
  const closed = accumulateTrainingGradient(assets, open.reader, [item(assets, [0, 1], {exposure: 'tight'})], {
    profile: 'rank6',
    exposure: 'tight',
  });
  assert.equal(closed.admitted, 0);
  assert.equal(closed.blocked[0].reason, 'exposure-mismatch');
});

test('同一向量被多个视图读取时梯度累积而不是覆盖', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');
  const result = accumulateTrainingGradient(
    assets,
    reader,
    [
      item(assets, [0, 1]),
      item(assets, [5, 2]),
      item(assets, [0, 1, 5]), // triple also writes slot 20 via fields 0 and 5
    ],
    {profile: 'rank4', exposure: 'open'},
  );
  assert.equal(result.admitted, 3);
  const shared = result.views.find((entry) => entry.slot === 20);
  assert.ok(shared);
  assert.equal(shared.assembled ?? shared.accumulated, true);
  assert.ok(shared.objectives.length >= 2);

  // 手工重算 slot 20 的梯度，验证是求和而非最后一个视图的覆盖。
  const manual = new Map();
  for (const targets of [
    {kind: 'pair', fields: [0, 1], upstream: 1},
    {kind: 'pair', fields: [5, 2], upstream: 1},
    {kind: 'triple', fields: [0, 1, 5], upstream: 1},
  ]) {
    const readout = reader.embedding(0) && null;
    void readout;
    const entry = result.itemResults.find((candidate) =>
      candidate.item.endsWith(`${targets.kind}:${targets.fields.join(':')}`),
    );
    accumulateGradient(manual, entry.gradient);
  }
  const bySlot = manual.get(20);
  const accumulated = result.gradient.get(20);
  assert.ok(bySlot && accumulated);
  for (let i = 0; i < assets.dimension; i += 1) {
    assert.ok(Math.abs(bySlot[i] - accumulated[i]) < 1e-12);
  }
});

test('经过共享槽的梯度按物理槽累加，别名两端看到同一坐标', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');
  const result = accumulateTrainingGradient(assets, reader, [item(assets, [0, 1]), item(assets, [5, 2])], {
    profile: 'rank4',
    exposure: 'open',
  });
  const views = result.views.map((entry) => entry.slot).sort((a, b) => a - b);
  // 0 与 5 都写 slot 20；slot 1 与 2 各来自一个域。
  assert.ok(views.includes(20));
  assert.equal(views.filter((slot) => slot === 20).length, 1);
  assert.deepEqual(result.views.find((entry) => entry.slot === 20).fields.sort(), [0, 5]);
});

test('提交档位只接受其合法轴的损失，新增轴上的损失被剔除', () => {
  const assets = syntheticAssets();
  const read48 = readerFor(assets, 'open');
  const highTier = [
    item(assets, [3, 4], {profile: 'rank6'}),
  ];
  const training = accumulateTrainingGradient(assets, read48.reader, highTier, {
    profile: 'rank6',
    exposure: 'open',
  });
  assert.deepEqual(training.itemResults[0].axes, [1, 2, 5]);
  assert.equal(training.itemResults[0].admitted, true);

  // 已有请求在两个档位共享读，但 rank4 只合法 0..3。
  const restricted = admissibleContributions(assets, training, {exposure: 'open', profile: 'rank4'});
  assert.equal(restricted.rejected.length, 0);
  assert.deepEqual(restricted.admitted[0].axes, [1, 2]);
  assert.deepEqual(restricted.admitted[0].droppedAxes, [5]);
  // 低档梯度只含保留的轴。
  for (const vector of restricted.gradient.values()) {
    assert.equal(vector[5], 0);
  }
});

test('域切换后的梯度来源保持不变，不被改写成新政策身份', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');
  const stale = createTrainingItem(assets, {
    kind: 'pair',
    fields: [0, 1],
    profile: 'rank4',
    exposure: 'open',
  });
  const training = accumulateTrainingGradient(assets, reader, [stale], {
    profile: 'rank4',
    exposure: 'open',
  });
  assert.equal(training.itemResults[0].exposure, 'open');
  const admissions = admissibleContributions(assets, training, {exposure: 'tight', profile: 'rank4'});
  assert.equal(admissions.rejected[0].reason, 'stale-exposure');
  assert.equal(admissions.rejected[0].exposure, 'open');
  // 合法分量仍然可以被算出来（供提案报告），但不会被当作 tight 的损失重用。
  assert.equal(admissions.gradient.size, 0);
});

test('没有合同的三元组或数量不符的项被拒绝', () => {
  const assets = syntheticAssets();
  assert.throws(() => item(assets, [0, 2]), /no published pair contract/);
  const {reader} = readerFor(assets, 'open');
  const wrongKind = {
    ...item(assets, [0, 1]),
    kind: 'triple',
  };
  const result = accumulateTrainingGradient(assets, reader, [wrongKind], {
    profile: 'rank4',
    exposure: 'open',
  });
  assert.equal(result.admitted, 0);
  assert.equal(result.blocked[0].reason, 'bad-arity');
});

test('termGradient 只对合法轴产生分量', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');
  const training = accumulateTrainingGradient(assets, reader, [item(assets, [0, 1])], {
    profile: 'rank4',
    exposure: 'open',
  });
  const slots = [...training.gradient.keys()].sort((a, b) => a - b);
  assert.deepEqual(slots, [1, 20]);
  const slot20 = training.gradient.get(20);
  for (const axis of [2, 4, 5]) assert.equal(slot20[axis], 0);
  for (const axis of [0, 1, 3]) assert.notEqual(slot20[axis], 0);
  // 检查手算：d/dv0[axis] = v1[axis]，slot 20 权重 1。
  const v1 = reader.embedding(1).vector;
  for (const axis of [0, 1, 3]) {
    assert.ok(Math.abs(slot20[axis] - v1[axis]) < 1e-12);
  }
  void termGradient;
});
