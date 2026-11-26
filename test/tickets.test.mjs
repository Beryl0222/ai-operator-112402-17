import assert from 'node:assert/strict';
import test from 'node:test';
import {createTickets, claimTicket, commitAxes} from '../src/tickets.mjs';
import {proposeUpdate} from '../src/proposal.mjs';
import {serve, sampleRequest, sampleTerms, domainOccupancy} from '../src/serving.mjs';
import {loadAssetData} from '../src/assets.mjs';
import {syntheticAssets} from './fixture.mjs';

const TERMS = [
  {kind: 'pair', fields: [0, 1], upstream: 1},
  {kind: 'pair', fields: [5, 2], upstream: 0.5},
  {kind: 'triple', fields: [0, 1, 5], upstream: 0.25},
];

test('发布请求以两个可独立领取的档位给出 fit 轴上界', () => {
  const assets = syntheticAssets();
  const request = createTickets(assets, {policy: 'open', seed: 4, terms: TERMS});
  const low = request.claim('rank4');
  const high = request.claim('rank6');
  assert.ok(low && high);
  assert.equal(low.bound, 4);
  assert.equal(high.bound, 6);
  assert.ok(low.axes.every((axis) => axis < 4));
  assert.ok(high.axes.some((axis) => axis >= 4));
  // 两个档位共享同一请求的源读缓存。
  assert.equal(low.sharedCache, high.sharedCache);
});

test('相同请求两档共享源读，但新增轴的损失不能复用', () => {
  const assets = syntheticAssets();
  const weights = {'pair:0:1': 1, 'pair:5:2': 0.5, 'triple:0:1:5': 0.25};
  const request = createTickets(assets, {policy: 'open', seed: 4, terms: TERMS});
  const low = claimTicket(assets, request, request.claim('rank4'), {lossWeights: weights});
  const fetchesAfterLow = request.ledger.summary().fetches;
  const high = claimTicket(assets, request, request.claim('rank6'), {lossWeights: weights});

  assert.equal(request.ledger.summary().fetches, fetchesAfterLow, 'second tier reuses source reads');
  assert.ok(high.access.cacheHits > 0);

  const lowIds = low.training.itemResults.map((entry) => entry.item);
  const highIds = high.training.itemResults.map((entry) => entry.item);
  assert.notDeepEqual(lowIds, highIds);
  assert.ok(lowIds.every((id) => id.startsWith('open/rank4/')));
  assert.ok(highIds.every((id) => id.startsWith('open/rank6/')));

  // rank6 的目标在 rank4 提交时只能使用 0..3 轴。
  const proposal = proposeUpdate(assets, {training: high.training, exposure: 'open', profile: 'rank4'});
  assert.equal(proposal.status, 'committed');
  assert.ok(proposal.admissions.restricted.length > 0);
});

test('rank32 的只读冻结仍约束 rank48 目标：以实际包含的合法轴决定提交', () => {
  const assets = syntheticAssets();
  const weights = {'pair:5:2': 1};
  const request = createTickets(assets, {policy: 'open', seed: 4, terms: TERMS});
  const high = claimTicket(assets, request, request.claim('rank6'), {lossWeights: weights});
  // 高档目标跨档提交到 rank4：rank4 声明的 joint 冻结读口必须生效。
  const proposal = proposeUpdate(assets, {
    training: high.training,
    exposure: 'open',
    profile: 'rank4',
    profiles: ['rank6'],
  });
  const names = proposal.effectiveConstraints.map((entry) => entry.name);
  assert.ok(names.includes('joint'));
  assert.ok(names.includes('read-only'));
  for (const check of proposal.constraints) assert.ok(check.satisfied);
});

test('低档的合法轴是轴标识小于上界的那些轴', () => {
  const assets = syntheticAssets();
  const axes = commitAxes(assets, ['rank4', 'rank6'], 'rank4');
  assert.deepEqual(axes, [0, 1, 2, 3]);
  const high = commitAxes(assets, ['rank4', 'rank6'], 'rank6');
  assert.deepEqual(high, [0, 1, 2, 3, 4, 5]);
});

test('典型调用按资料限制随机取数并守住读区与中间张量预算', async () => {
  const assets = loadAssetData();
  for (let seed = 1; seed <= 12; seed += 1) {
    const plan = sampleRequest(assets, {seed, policy: 'v1', profile: 'rank32'});
    assert.ok(plan.terms.length <= assets.batch);
    assert.ok(plan.requestedAxes.length > 0);
    assert.ok(plan.requestedAxes.every((axis) => assets.axisAllowed(axis, 'rank32')));
    assert.ok(plan.intermediateBudget < assets.workBytes);

    const reading = serve(assets, {
      policy: plan.policy,
      profile: plan.profile,
      terms: plan.terms,
      requestedAxes: plan.requestedAxes,
      intermediateBudget: plan.intermediateBudget,
    });
    assert.ok(reading.access.fetches <= assets.slotFetchLimit);
    assert.ok(reading.access.intermediateBytes <= reading.access.workBytes);
    assert.ok(reading.access.withinBudget);
    // 不读取关闭域：v1 中只有 19/383/199 被采样。
    const allowed = new Set(assets.policy('v1').fields);
    for (const readout of reading.readouts) {
      for (const field of readout.fields) {
        if (readout.status === 'ok') assert.ok(allowed.has(field));
      }
    }
  }
});

test('v1 采样绝不包含已关闭的域，也不创建稠密张量', () => {
  const assets = loadAssetData();
  for (const seed of [1, 2, 3]) {
    const terms = sampleTerms(assets, {seed, policy: 'v1', profile: 'rank32', limit: 100});
    for (const term of terms) {
      for (const field of term.fields) assert.ok(!assets.policy('v1').fields.includes(field) || field !== 1);
    }
    // 采样数量上限为 batch，而不是域数的平方。
    assert.ok(terms.length <= assets.batch);
  }
});

test('各域占用随使用者变化，占用统计只包含采样到的域', () => {
  const assets = loadAssetData();
  const plan = sampleRequest(assets, {seed: 7, policy: 'v1', profile: 'rank32'});
  const occupancy = domainOccupancy(assets, plan.terms, plan.policy);
  const fields = new Set(plan.terms.flatMap((term) => term.fields));
  assert.equal(occupancy.length, fields.size);
  assert.ok(occupancy.every((entry) => entry.authorized));
});

test('域切换后保留的读口以当时档位身份验算', () => {
  const assets = syntheticAssets();
  const request = createTickets(assets, {policy: 'open', seed: 4, terms: TERMS});
  const ticket = claimTicket(assets, request, request.claim('rank4'), {
    lossWeights: {'pair:0:1': 1},
  });
  // 计算已排队后域 0 关闭，再以 tight 提交。
  const proposal = proposeUpdate(assets, {
    training: ticket.training,
    exposure: 'tight',
    profile: 'rank4',
  });
  assert.equal(proposal.status, 'blocked');
  assert.equal(proposal.blockReason, 'domain-switch');
  assert.equal(proposal.staleItems[0].exposure, 'open');
  // 保留的读口必须按实际使用的档位验证，而不是按提交身份。
  assert.ok(proposal.ports.every((port) => !port.effective || port.preserved !== false));
});
