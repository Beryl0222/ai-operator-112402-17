import assert from 'node:assert/strict';
import test from 'node:test';
import {AccessLedger, SourceReader, auditCache} from '../src/access.mjs';
import {readTerm} from '../src/interactions.mjs';
import {frozenConstraints} from '../src/views.mjs';
import {DIM, syntheticAssets, slotVector} from './fixture.mjs';

function readerFor(assets, policy, extra = {}) {
  const ledger = new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: assets.workBytes,
    dimension: assets.dimension,
    policy,
  });
  return {ledger, reader: new SourceReader(assets, {ledger, policy, ...extra})};
}

test('交叉值是合法轴乘积之和，且从不建立稠密张量', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');
  const readout = readTerm(assets, reader, {kind: 'pair', fields: [0, 1]}, {profile: 'rank4'});
  assert.equal(readout.status, 'ok');
  assert.deepEqual(readout.axes, [0, 1, 3]);

  const a = reader.embedding(0).vector;
  const b = reader.embedding(1).vector;
  const expected = a[0] * b[0] + a[1] * b[1] + a[3] * b[3];
  assert.ok(Math.abs(readout.value - expected) < 1e-12);

  // Working memory is charged per legal axis, not per field pair.
  assert.ok(readout.axes.length <= assets.dimension);
  assert.equal(readout.embeddings.length, 2);
  assert.deepEqual(readout.embeddings.map((vector) => vector.length), [DIM, DIM]);
});

test('三重项只在三个域都授权时求值，且只用声明的轴', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');
  const readout = readTerm(assets, reader, {kind: 'triple', fields: [0, 1, 5]}, {profile: 'rank4'});
  assert.equal(readout.status, 'ok');
  assert.deepEqual(readout.axes, [0, 1]);
  // 轴 4 被档位裁掉，其余轴没有声明，系数为零。
  assert.deepEqual(readout.dropped, [{axis: 4, reason: 'beyond-rank'}]);
});

test('关闭的域不会触发任何读槽，即使共享参数看起来可用', () => {
  const assets = syntheticAssets();
  const {reader, ledger} = readerFor(assets, 'tight');
  const readout = readTerm(assets, reader, {kind: 'pair', fields: [0, 1]}, {profile: 'rank4'});
  assert.equal(readout.status, 'unavailable');
  assert.equal(readout.reason, 'unauthorized');
  assert.equal(ledger.calls, 0);
  assert.equal(ledger.fetches, 0);
  // 未授权域连共享槽 20 也不读。
  assert.equal(reader.slot(20, {field: 0}).reason, 'unauthorized');
});

test('已许可的别名仍可读公共槽，但不因此获得关闭域的新向量', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'tight');
  const viaAlias = reader.slot(20, {field: 5});
  assert.equal(viaAlias.available, true);
  assert.deepEqual([...viaAlias.value], [...slotVector(20)]);
  // 域 5 不能假扮拥有域 0 的私有槽映射。
  assert.equal(reader.slot(20, {field: 5, requireOwner: true}).available, true);
  assert.equal(reader.slot(1, {field: 5, requireOwner: true}).reason, 'not-an-owner');
});

test('政策收紧后缓存的关闭域身份只可核对、不可重读', () => {
  const assets = syntheticAssets();
  const cache = new Map();
  const open = readerFor(assets, 'open', {sharedCache: cache});
  open.reader.embedding(0);
  open.reader.embedding(3);
  assert.ok(cache.size >= 2);

  const tight = readerFor(assets, 'tight', {sharedCache: cache});
  // 域 0 的私有槽 20 在缓存里，但关闭域不得再读。
  assert.equal(tight.reader.slot(20, {field: 0}).reason, 'unauthorized');
  // 共享槽的公开身份可被核对。
  const audit = auditCache(assets, cache, 'open', 'tight');
  assert.deepEqual(audit.closed.sort(), [0, 3, 4]);
  assert.ok(audit.held.some((entry) => entry.key === 20));
  assert.ok(audit.reachable.some((entry) => entry.key === 20));
  assert.deepEqual(audit.aliasShared, [
    {key: 20, closed: 0, open: 5},
    {key: 21, closed: 3, open: 6},
  ]);
});

test('真实缺失与低秩后恰为零的输出分开报告', () => {
  const assets = syntheticAssets();
  const {reader} = readerFor(assets, 'open');

  const missing = readTerm(assets, reader, {kind: 'pair', fields: [0, 1]}, {
    profile: 'rank4',
    requestedAxes: [5],
  });
  assert.equal(missing.status, 'ok');
  assert.equal(missing.applicable, true);
  assert.equal(missing.zero, true);
  assert.equal(missing.zeroKind, 'low-rank');
  assert.deepEqual(missing.axes, []);
  assert.notEqual(missing.reason, 'unavailable');

  const unavailable = readTerm(assets, reader, {kind: 'pair', fields: [0, 3]}, {profile: 'rank4'});
  assert.equal(unavailable.status, 'unavailable');
  assert.equal(unavailable.reason, 'no-contract');

  const closed = readerFor(assets, 'tight');
  const closedReadout = readTerm(assets, closed.reader, {kind: 'pair', fields: [0, 1]}, {profile: 'rank4'});
  assert.equal(closedReadout.zero, null);
  assert.equal(closedReadout.value, null);
});

test('触发读槽按 (table,version,key) 计一次，观测记录目的域', () => {
  const assets = syntheticAssets();
  const {reader, ledger} = readerFor(assets, 'open');
  readTerm(assets, reader, {kind: 'pair', fields: [0, 1]}, {profile: 'rank4'});
  const first = ledger.summary();
  assert.equal(first.fetches, 2); // slot 20 (shared by 0) + slot 1
  assert.ok(first.calls >= 2);
  assert.ok(ledger.observations.every((entry) => Number.isInteger(entry.field)));
  assert.deepEqual(
    ledger.observations.map((entry) => [entry.key, entry.field]),
    [
      [20, 0],
      [1, 1],
    ],
  );
  // 同一请求第二次读同一槽只算一次源取数。
  reader.embedding(0);
  assert.equal(ledger.summary().fetches, first.fetches);
});

test('读槽调用预算耗尽后停止服务，并按调用记录记录目的域', () => {
  const assets = syntheticAssets();
  const ledger = new AccessLedger({
    slotFetchLimit: 1,
    workBytes: assets.workBytes,
    dimension: assets.dimension,
  });
  const reader = new SourceReader(assets, {ledger, policy: 'open'});
  const first = reader.slot(20, {field: 0});
  assert.equal(first.available, true);
  assert.deepEqual(ledger.observations[0], {
    index: 1,
    key: 20,
    field: 0,
    purpose: 'read',
    cached: false,
    requirement: 'shared',
  });
  const second = reader.slot(1, {field: 1});
  assert.equal(second.available, false);
  assert.equal(second.reason, 'slot-call-limit');
  assert.equal(ledger.withinBudget, false);
});

test('中间张量预算以坐标字节计，超出后拒绝继续求值', () => {
  const assets = syntheticAssets();
  const ledger = new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: 8,
    dimension: assets.dimension,
  });
  assert.equal(ledger.reserveIntermediate(8, 'axes').ok, true);
  const denied = ledger.reserveIntermediate(8, 'axes');
  assert.equal(denied.ok, false);
  assert.equal(denied.reason, 'work-bytes');
  assert.equal(ledger.withinBudget, false);
});
