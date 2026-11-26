import assert from 'node:assert/strict';
import test from 'node:test';
import {AccessLedger, SourceReader} from '../src/access.mjs';
import {frozenConstraints, mixtureDirection, portValue} from '../src/views.mjs';
import {projectStep, sparseDot, sparseNorm, validateConstraints} from '../src/linear.mjs';
import {proposeUpdate} from '../src/proposal.mjs';
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

function gradientAt(assets, slots, scale = 1) {
  const gradient = new Map();
  for (const slot of slots) {
    gradient.set(
      slot,
      Float64Array.from({length: assets.dimension}, (_, axis) => scale * Math.cos(0.4 * (slot + axis + 1))),
    );
  }
  return gradient;
}

test('protectedMixtures 先由物理槽解释，再按映射与系数合成读口方向', () => {
  const assets = syntheticAssets();
  const mixture = assets.protectedMixtures.find((entry) => entry.name === 'joint');
  const {direction, axes} = mixtureDirection(assets, mixture);
  assert.deepEqual(axes, [0, 1, 2, 0, 1, 2]);

  // 手工：成员 5 (slot 20, 权 1, 系数 1) 与成员 2 (slot 2, 权 1, 系数 -0.5)。
  const weights = mixture.weights;
  const slot20 = direction.get(20);
  const slot2 = direction.get(2);
  assert.ok(Math.abs(slot20[0] - weights[0]) < 1e-12);
  assert.ok(Math.abs(slot20[1] - weights[1]) < 1e-12);
  assert.ok(Math.abs(slot2[0] - -0.5 * weights[0]) < 1e-12);
  assert.ok(Math.abs(slot2[2] - -0.5 * weights[2]) < 1e-12);

  const base = new Map([
    [20, Float64Array.from([1, 2, 3, 4, 5, 6])],
    [2, Float64Array.from([7, 8, 9, 10, 11, 12])],
  ]);
  const value = portValue(direction, base);
  const expected = 1 * 1 + 2 * 2 + 1 * 3 - 0.5 * (1 * 7 + 2 * 8 + 1 * 9);
  assert.ok(Math.abs(value - expected) < 1e-12);
});

test('多个冻结视图共享别名时线性等式相依，投影一次满足全部', () => {
  const assets = syntheticAssets();
  const {constraints} = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  const raw = gradientAt(assets, [20, 2, 1, 3]);
  const projection = projectStep(raw, constraints);
  assert.equal(projection.invalid.length, 0);
  for (const residual of projection.residuals) {
    assert.ok(Math.abs(residual.dot) < 1e-10, `${residual.name} residual ${residual.dot}`);
  }
  const checks = validateConstraints(projection.delta, constraints);
  assert.ok(checks.every((check) => check.satisfied));

  // 投影只去掉与冻结方向相交的分量。
  assert.ok(projection.projectedNorm <= projection.rawNorm + 1e-12);
  assert.ok(projection.projectedNorm > 0);
});

test('逐域清零会丢掉仍可用的个性化变化；联合投影保留它', () => {
  const assets = syntheticAssets();
  const {constraints} = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  // 关掉域 0 后，slot 20 仍由域 5 可写。
  const raw = gradientAt(assets, [20, 2], 0.5);
  const joint = projectStep(raw, constraints);

  // 朴素做法：逐域清零把 slot 20 整个删掉。
  const naive = new Map(raw);
  naive.delete(20);
  assert.ok(sparseNorm(naive) < sparseNorm(joint.delta));

  // 联合投影仍可能保留 slot 20 的某些方向，同时不改变冻结视图。
  const readOnly = constraints.find((constraint) => constraint.name === 'read-only');
  assert.ok(Math.abs(sparseDot(readOnly.direction, joint.delta)) < 1e-10);
  assert.ok(joint.delta.has(20));
});

test('提案在域切换时给出可应用分量与被阻止的关系', () => {
  const assets = syntheticAssets();
  const reader = readerFor(assets, 'tight');
  const training = {
    exposure: 'open',
    profile: 'rank4',
    itemResults: [
      {
        item: 'open/rank4/pair:0:1',
        admitted: true,
        exposure: 'open',
        profile: 'rank4',
        axes: [0, 1, 3],
        gradient: gradientAt(assets, [20, 1]),
        kind: 'pair',
        fields: [0, 1],
      },
    ],
  };
  const proposal = proposeUpdate(assets, {training, exposure: 'tight', profile: 'rank4'});
  assert.equal(proposal.status, 'blocked');
  assert.equal(proposal.blockReason, 'domain-switch');
  assert.equal(proposal.applicable.componentCount, 0);
  assert.ok(proposal.blockedRelations.some((entry) => entry.relation === 'gradient-source'));
  assert.ok(proposal.staleItems.length > 0);
  assert.ok(proposal.fallback);
  void reader;
});

test('冻结约束在档位之间生效：rank4 提交仍检查声明于 rank6 的读口', () => {
  const assets = syntheticAssets();
  const low = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  const high = frozenConstraints(assets, {policy: 'open', profile: 'rank6'});
  const names = new Set(low.constraints.map((constraint) => constraint.name));
  assert.ok(names.has('joint'));
  assert.ok(names.has('deep'));

  // 声明于 rank6 的读口只用到 0..3，因此在 rank4 提交同样是可读约束。
  const deep = low.constraints.find((constraint) => constraint.name === 'deep');
  assert.deepEqual([...deep.axes].sort((a, b) => a - b), [0, 1, 3]);
  assert.ok(high.constraints.some((constraint) => constraint.name === 'deep'));
});

test('某档位合法域生成的冻结等式只作用于该档位实际包含的轴', () => {
  const assets = syntheticAssets();
  const {constraints} = frozenConstraints(assets, {
    policy: 'tight',
    profile: 'rank6',
    profiles: ['rank4'],
  });
  // 域 0 与 3 关闭，但 joint(5,2) 与 deep(1) 的成员都仍许可。
  assert.ok(constraints.some((constraint) => constraint.name === 'joint'));
  assert.ok(constraints.some((constraint) => constraint.name === 'deep'));
  // 读口方向只能触碰关闭域仍然持有的共享槽 20 —— 通过许可别名 5 读取。
  const joint = constraints.find((constraint) => constraint.name === 'joint');
  assert.ok(joint.fields.includes(5));
  for (const axis of joint.axes) {
    assert.ok(assets.eligibleAxis.has(axis));
  }
});

test('成员域关闭时联合读口变成被阻止关系，而不是被截断的近似', () => {
  const assets = syntheticAssets({
    source: {
      protectedMixtures: [
        [
          'crosses-closed',
          'rank4',
          [
            [0, 1, [0, 1, 2]],
            [5, -1, [0, 1, 2]],
          ],
          [1, 1, 1],
        ],
      ],
    },
  });
  const {constraints, blocked, ports} = frozenConstraints(assets, {policy: 'tight', profile: 'rank4'});
  assert.ok(!constraints.some((constraint) => constraint.name === 'crosses-closed'));
  const blockedPort = blocked.find((entry) => entry.name === 'crosses-closed');
  assert.equal(blockedPort.reason, 'domain-switch');
  assert.deepEqual(blockedPort.closedFields, [0]);
  // 但读口本身仍然被列出，供推荐端核对身份。
  assert.ok(ports.some((port) => port.name === 'crosses-closed' && port.readable === false));
});

test('冻结视图中两个许可域的贡献可以相互抵消', () => {
  const assets = syntheticAssets();
  const {constraints} = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  const joint = constraints.find((constraint) => constraint.name === 'joint');
  // 构造 Δ：只在 slot 20 与 slot 2 上，使联合视图变化为零但谁也不为零。
  const delta = new Map();
  const d20 = new Float64Array(assets.dimension);
  const d2 = new Float64Array(assets.dimension);
  d20[0] = 1;
  d2[0] = 2;
  delta.set(20, d20);
  delta.set(2, d2);

  const change = sparseDot(joint.direction, delta);
  assert.ok(Math.abs(change) < 1e-12);
  assert.equal(d20[0], 1);
  assert.equal(d2[0], 2);
});

test('不可读的格点越权读口作为被阻止关系报告', () => {
  const assets = syntheticAssets({
    source: {
      protectedMixtures: [
        [
          'overreach',
          'rank4',
          [
            [5, 1, [0, 1, 9]], // 轴 9 超出维度
          ],
          [1, 0, 0],
        ],
      ],
    },
  });
  const {constraints, blocked} = frozenConstraints(assets, {policy: 'open', profile: 'rank4'});
  assert.ok(!constraints.some((constraint) => constraint.name === 'overreach'));
  assert.ok(blocked.some((entry) => entry.name === 'overreach' && entry.reason === 'beyond-lattice'));
});
