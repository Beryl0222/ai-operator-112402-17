// On-device serving: sample a request under the dataset's own budgets and run
// the forward pass over the contracts that still apply.
//
// The sampler never invents fields beyond `fields`, never reads an axis outside
// the tier's read region, and stops before the slot-fetch or working-memory
// limits.  It may skip terms whose domains are closed: "共享参数不等于共享授权"
// means an alias does not make a closed field's data available again.

import {createRandom} from './assets.mjs';
import {AccessLedger, SourceReader} from './access.mjs';
import {accumulateGradient, readRegion, readTerm} from './interactions.mjs';
import {accumulateTrainingGradient} from './training.mjs';
import {proposeUpdate, snapshotReads} from './proposal.mjs';
import {readPorts} from './views.mjs';

export function sampleTerms(assets, options = {}) {
  const {seed = 1, policy = assets.latestPolicyName, profile = 'rank32', limit = assets.batch, kinds = ['pair', 'triple']} =
    options;
  const rng = createRandom(seed);
  const allowed = assets.policy(policy)?.fieldSet ?? new Set();
  const candidates = [];
  for (const contract of [...assets.pairs, ...assets.triples]) {
    if (!kinds.includes(contract.kind)) continue;
    if (!contract.fields.every((field) => allowed.has(field))) continue;
    const {legal} = assets.legalAxes(contract.axes, profile);
    if (!legal.length) continue;
    candidates.push(contract);
  }
  const chosen = rng.shuffle(candidates).slice(0, Math.max(0, limit));
  return chosen.map((contract, index) => ({
    kind: contract.kind,
    fields: [...contract.fields],
    profile,
    exposure: policy,
    upstream: Number((0.125 * (index + 1)).toFixed(6)),
  }));
}

export function sampleAxes(assets, profile, options = {}) {
  const {seed = 1, count = null} = options;
  const region = readRegion(assets, profile);
  if (count === null || count >= region.length) return region;
  const rng = createRandom(seed + 7919);
  return rng.shuffle(region).slice(0, count).sort((a, b) => a - b);
}

/**
 * Sample a whole request shape within the dataset's limits.
 *
 * Request shape and per-domain occupancy vary by user, so the caller supplies
 * only a seed: the term count stays within `batch`, terms are weighted by how
 * many authorized contracts a domain pair actually has, the read region stays
 * inside the tier, and the intermediate budget stays below `workBytes`.
 */
export function sampleRequest(assets, options = {}) {
  const {
    seed = 1,
    policy = assets.latestPolicyName,
    profile = 'rank32',
    maxTerms = assets.batch,
    axisFraction = null,
    budgetFraction = null,
  } = options;
  const rng = createRandom(seed * 2654435761);
  const allowed = assets.policy(policy)?.fieldSet ?? new Set();
  const contracts = [...assets.pairs, ...assets.triples].filter(
    (contract) =>
      contract.fields.every((field) => allowed.has(field)) && assets.legalAxes(contract.axes, profile).legal.length > 0,
  );
  const limit = Math.max(0, Math.min(maxTerms, contracts.length, assets.batch));
  const chosen = rng.shuffle(contracts).slice(0, limit);
  const region = readRegion(assets, profile);
  const fraction = axisFraction ?? 0.4 + rng.next() * 0.6;
  const axisCount = Math.max(1, Math.min(region.length, Math.round(region.length * fraction)));
  const requestedAxes = rng.shuffle(region).slice(0, axisCount).sort((a, b) => a - b);
  const budget = budgetFraction ?? 0.25 + rng.next() * 0.75;
  return {
    policy,
    profile,
    terms: chosen.map((contract, index) => ({
      kind: contract.kind,
      fields: [...contract.fields],
      profile,
      exposure: policy,
      upstream: Number((0.125 * (index + 1)).toFixed(6)),
      requestedAxes,
    })),
    requestedAxes,
    intermediateBudget: Math.max(1024, Math.floor(assets.workBytes * budget)),
    occupancy: domainOccupancy(assets, chosen, policy),
  };
}

/** How often each domain appears in the sampled request. */
export function domainOccupancy(assets, terms, policy = assets.latestPolicyName) {
  const counts = new Map();
  for (const term of terms) {
    for (const field of term.fields) counts.set(field, (counts.get(field) ?? 0) + 1);
  }
  const allowed = assets.policy(policy)?.fieldSet ?? new Set();
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([field, count]) => ({field, count, authorized: allowed.has(field)}));
}

/**
 * Run a forward pass over sampled (or caller-supplied) terms.
 *
 * `intermediateBudget` lets the caller shrink the batch when a device reports
 * less working memory than the dataset's `workBytes`; the per-term coordinate
 * footprint is tiny, so this is a bound on *summed* intermediates, never a
 * dense tensor.
 */
export function serve(assets, options = {}) {
  const {
    policy = assets.latestPolicyName,
    profile = 'rank32',
    seed = 1,
    limit = assets.batch,
    terms = null,
    requestedAxes = null,
    sharedCache = null,
    purpose = 'score',
    ledger: providedLedger = null,
    intermediateBudget = assets.workBytes,
  } = options;

  const ledger = providedLedger ?? new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: Math.min(assets.workBytes, intermediateBudget),
    dimension: assets.dimension,
    policy,
  });
  const reader = new SourceReader(assets, {ledger, policy, purpose, sharedCache, profile});
  const selected = terms ?? sampleTerms(assets, {seed, policy, profile, limit});

  const readouts = [];
  const skipped = [];
  let served = 0;
  for (const term of selected) {
    const readout = readTerm(
      assets,
      reader,
      {...term, purpose: term.purpose ?? purpose},
      {profile: term.profile ?? profile, requestedAxes: term.requestedAxes ?? requestedAxes},
    );
    if (!readout.applicable) {
      skipped.push({
        id: readout.id,
        reason: readout.reason,
        status: readout.reason === 'unauthorized' ? 'blocked' : 'unavailable',
        missingFields: readout.missingFields,
      });
      readouts.push(readout);
      continue;
    }
    // Coordinate footprint of one term: one scalar per legal axis plus the
    // participating embeddings, charged as bytes, never as a dense tensor.
    const axesUsed = readout.axes.length;
    const bytes = axesUsed * 8 + readout.fields.length * axesUsed * 8;
    const reserved = ledger.reserveIntermediate(bytes, readout.id);
    if (!reserved.ok) {
      skipped.push({id: readout.id, reason: reserved.reason, status: 'skipped'});
      continue;
    }
    served += 1;
    readouts.push(readout);
  }

  return {
    policy,
    profile,
    readouts,
    skipped,
    ports: readPorts(assets, reader),
    ledger,
    access: ledger.summary(),
    served,
    requested: selected.length,
  };
}

/**
 * Combined step used by the recommender: a forward pass plus the backward pass
 * for the terms the device is allowed to learn from.
 *
 * `lossWeights` maps a term id to its released loss weight; a missing weight
 * means the term was not released for training, so it is read but not learned.
 */
export function serveAndLearn(assets, options = {}) {
  const {
    lossWeights = {},
    published = null,
    seed = 1,
    profile = 'rank32',
    policy = assets.latestPolicyName,
    requestedAxes = null,
    sharedCache = null,
    limit = assets.batch,
  } = options;

  const ledger = new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: assets.workBytes,
    dimension: assets.dimension,
    policy,
  });
  const reader = new SourceReader(assets, {ledger, policy, purpose: 'loss', sharedCache, profile});
  const selected = options.terms ?? sampleTerms(assets, {seed, policy, profile, limit});

  const items = [];
  for (const term of selected) {
    const id = `${term.kind}:${term.fields.join(':')}`;
    if (!(id in lossWeights)) continue;
    const allowedToPublish = published === null ? true : published(id);
    if (!allowedToPublish) continue;
    items.push({
      id,
      termId: id,
      kind: term.kind,
      fields: term.fields,
      profile: term.profile ?? profile,
      exposure: term.exposure ?? policy,
      published: true,
      lossWeight: lossWeights[id],
      upstream: term.upstream ?? 1,
      requestedAxes: term.requestedAxes ?? requestedAxes,
      purpose: 'loss',
    });
  }

  const training = accumulateTrainingGradient(assets, reader, items, {
    profile,
    exposure: policy,
    requestedAxes,
  });

  return {
    policy,
    profile,
    reader,
    ledger,
    training,
    gradient: training.gradient,
    access: ledger.summary(),
  };
}

/**
 * One end-to-end device call: sample a request within the published limits,
 * serve the applicable terms, optionally learn, and hold the resulting
 * proposal against the last legal read.
 *
 * The returned object is JSON-friendly apart from `ledger`/`reader`; the port
 * snapshot in `snapshot` is exactly what a blocked proposal hands back so the
 * recommender can keep using the previous readout.
 */
export function serveRequest(assets, options = {}) {
  const {
    seed = 1,
    policy = assets.latestPolicyName,
    profiles = null,
    tiers = profiles ?? ['rank32'],
    lossWeights = null,
    published = null,
    allowPartial = false,
    lastLegalRead = null,
    request = null,
  } = options;

  const plan = request ?? sampleRequest(assets, {seed, policy, profile: tiers[tiers.length - 1]});
  const reading = serve(assets, {
    policy: plan.policy,
    profile: plan.profile,
    terms: plan.terms,
    requestedAxes: plan.requestedAxes,
    intermediateBudget: plan.intermediateBudget,
  });
  const snapshot = snapshotReads(reading);

  let training = null;
  let proposal = null;
  if (lossWeights && Object.keys(lossWeights).length) {
    const learn = serveAndLearn(assets, {
      policy: reading.policy,
      profile: reading.profile,
      terms: plan.terms,
      requestedAxes: plan.requestedAxes,
      lossWeights,
      published,
    });
    training = learn.training;
    proposal = proposeUpdate(assets, {
      training,
      exposure: reading.policy,
      profile: reading.profile,
      profiles: tiers.filter((tier) => tier !== reading.profile),
      lastLegalRead: lastLegalRead ?? snapshot,
      allowPartial,
    });
  }

  return {
    plan,
    reading,
    snapshot,
    training,
    proposal,
    access: reading.access,
    occupancy: plan.occupancy,
  };
}

export {accumulateGradient, accumulateTrainingGradient};
