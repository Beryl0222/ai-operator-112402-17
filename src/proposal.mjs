// Turning a raw personalized gradient into a committable update.
//
// The legal step is the Euclidean projection of the raw gradient step onto the
// intersection of every frozen read view effective at the tier the commit is
// actually made under.  Constraints are joint over slots, not per field: after
// a domain exits and keeps a shared vector read-only, the fields that are still
// authorized may move that vector as long as their joint read view stays put,
// which usually means two domains move it in opposite directions.
//
// Failure modes are reported separately:
//   * a component whose source was read under a different exposure than the
//     commit (a domain switch) cannot be applied;
//   * a frozen relation that can no longer be read or verified (its member
//     domain is closed, or its axes sit outside the commit tier) blocks the
//     commit.
//
// When a commit is blocked the caller receives the components that *would* be
// applicable plus the blocked relations, and a snapshot of the last legal read
// so serving can keep using the previous readout.

import {
  projectStep,
  sparseDot,
  sparseNorm,
  sparseSupport,
  validateConstraints,
  weightedNorm,
} from './linear.mjs';
import {admissibleContributions} from './training.mjs';
import {frozenConstraints, mixtureDirection, readPorts, viewDirection} from './views.mjs';

export const DEFAULT_TOLERANCE = 1e-9;

function portDirections(assets, ports, effective) {
  const effectiveNames = new Set(effective.map((constraint) => `${constraint.kind}:${constraint.name}`));
  return ports.map((port) => {
    let direction = port.direction ?? null;
    if (!direction && port.kind === 'view') direction = viewDirection(assets, assets.view(port.name));
    if (!direction && port.kind === 'mixture') {
      direction = mixtureDirection(assets, assets.protectedMixtures.find((entry) => entry.name === port.name))
        ?.direction;
    }
    return {
      name: port.name,
      kind: port.kind,
      profile: port.profile ?? null,
      axes: port.axes ?? null,
      frozen: port.frozen !== false,
      readable: port.readable ?? null,
      effective: effectiveNames.has(`${port.kind}:${port.name}`),
      direction: direction ?? null,
    };
  });
}

function changeFor(direction, delta) {
  if (!direction) return null;
  return sparseDot(direction, delta);
}

/**
 * @param {object} assets
 * @param {object} options
 * @param {Map}    [options.gradient]        raw gradient (slot -> Float64Array)
 * @param {object} [options.training]        result of accumulateTrainingGradient
 * @param {string} options.exposure          policy identity the commit runs under
 * @param {string} options.profile           release tier of the commit
 * @param {string[]} [options.profiles]      every tier the target actually spans
 * @param {number} [options.lossWeight]
 * @param {number} [options.hNorm]           incoming H-norm from the trainer
 * @param {Array|Map} [options.hWeights]     per-axis H weights, reports the
 *                                           H-norm of the legal step
 * @param {object} [options.lastLegalRead]   snapshot to keep when blocked
 * @param {boolean} [options.allowPartial]   commit the applicable components
 *                                           even when relations are blocked
 */
export function proposeUpdate(assets, options) {
  const {
    gradient = null,
    training = null,
    exposure,
    profile,
    profiles = null,
    lossWeight = null,
    hNorm = null,
    hWeights = null,
    lastLegalRead = null,
    allowPartial = false,
    tolerance = DEFAULT_TOLERANCE,
  } = options;

  if (!assets.policy(exposure)) throw new Error(`unknown policy: ${exposure}`);
  const tier = assets.profile(profile);
  if (!tier) throw new Error(`unknown profile: ${profile}`);
  if (!gradient && !training) throw new Error('proposeUpdate needs a gradient or a training result');

  const spans = [
    ...new Set([
      profile,
      ...(profiles ?? []),
      ...((training?.itemResults ?? []).filter((item) => item.admitted).map((item) => item.profile)),
    ]),
  ];
  const blockedRelations = [];
  const staleItems = [];
  let raw;
  let admissions = null;

  if (training) {
    admissions = admissibleContributions(assets, training, {exposure, profile});
    raw = admissions.gradient;
    for (const rejected of admissions.rejected) {
      staleItems.push(rejected);
      blockedRelations.push({
        relation: 'gradient-source',
        kind: 'loss',
        item: rejected.item,
        reason: rejected.reason,
        exposure: rejected.exposure ?? training.exposure ?? null,
        axes: rejected.axes ?? null,
      });
    }
  } else {
    raw = gradient;
  }

  // A slot with no authorized alias cannot carry an update: the fields that are
  // still authorized never wrote to it.  This is the alias rule, checked at the
  // physical-slot level rather than per field.
  const policyFields = assets.policy(exposure).fieldSet;
  const writable = new Map();
  for (const [slot, vector] of raw) {
    const owners = assets.fieldsForSlot(slot);
    if (owners.some((field) => policyFields.has(field))) writable.set(slot, vector);
    else {
      blockedRelations.push({
        relation: 'shared-slot',
        kind: 'slot',
        slot,
        fields: owners,
        reason: 'alias-frozen',
        detail: 'no authorized alias owns this slot',
      });
    }
  }

  const rawNorm = sparseNorm(writable);
  const {
    constraints,
    ports,
    blocked,
    epsilon,
  } = frozenConstraints(assets, {policy: exposure, profiles: spans, profile});

  // The projection is joint over slots and is the only correct answer for a
  // shared alias held by a closed domain: per-field zeroing would discard the
  // still-authorized domain's personalized change, while touching the common
  // slot alone would move the frozen read view.  Keeping the component only
  // inside the intersection's null space keeps both requirements.
  const projection = projectStep(writable, constraints, {tolerance: 1e-12});
  const checks = validateConstraints(projection.delta, constraints, {epsilon});
  const hNormValue = hWeights ? weightedNorm(projection.delta, hWeights) : null;

  for (const entry of blocked) {
    blockedRelations.push({relation: 'read-port', ...entry});
  }

  const violations = checks.filter((check) => !check.satisfied);

  // Only a frozen relation that cannot even be observed at this exposure is a
  // hard blocker.  A relation outside the spanned tiers was never part of the
  // released region, so it is informational.
  const hardBlockers = blockedRelations.filter(
    (entry) => entry.reason === 'domain-switch' || entry.reason === 'closed-slot',
  );

  // A joint-cancellation slot is a *blocked relation*, not a blocked commit:
  // the rest of the personalized update is still applicable.
  const blockedBy = staleItems.length
    ? 'domain-switch'
    : hardBlockers.length
      ? 'frozen-read-port'
      : projection.invalid.length
        ? 'invalid-frozen-view'
        : violations.length
          ? 'constraint-violation'
          : null;

  const blockedHard = blockedBy !== null;

  const components = sparseSupport(projection.delta);
  const rawComponents = sparseSupport(writable);
  const appliedPorts = readPortsForUpdate(assets, projection.delta, ports, constraints, tolerance);

  const applicable = {
    delta: projection.delta,
    components,
    componentCount: components.reduce((total, entry) => total + entry.dims.length, 0),
    stepNorm: projection.projectedNorm,
  };

  // With `allowPartial` the applicable components may still be written while
  // the blocked relations are reported and the affected ports keep the last
  // legal read.  Without it, nothing is written.
  const status =
    blockedHard && !(allowPartial && !violations.length && applicable.componentCount > 0)
      ? 'blocked'
      : blockedHard
        ? 'partial'
        : 'committed';

  return {
    status,
    applied: status !== 'blocked',
    blockReason: blockedBy,
    written: status === 'partial' ? applicable.delta : status === 'committed' ? applicable.delta : new Map(),
    exposure,
    profile,
    profiles: spans,
    tier: {bound: tier.bound, epsilon},
    delta: projection.delta,
    components,
    componentCount: applicable.componentCount,
    rawComponents,
    rawComponentCount: rawComponents.reduce((total, entry) => total + entry.dims.length, 0),
    rawNorm,
    stepNorm: projection.projectedNorm,
    hNorm: hNormValue ?? hNorm,
    removedNorm: projection.removedNorm,
    // Always expose the applicable vector, even when blocked, so a caller can
    // inspect what would have been written before deciding to fall back.
    applicable,
    lossWeight,
    constraints: checks,
    effectiveConstraints: constraints.map((constraint) => ({
      name: constraint.name,
      kind: constraint.kind,
      profile: constraint.profile,
      axes: constraint.axes,
    })),
    dependentConstraints: projection.dependent,
    invalidConstraints: projection.invalid,
    blockedRelations: dedupeRelations(blockedRelations),
    staleItems,
    admissions,
    ports: appliedPorts,
    fallback: status === 'blocked' ? buildFallback(lastLegalRead, blockedBy, blockedRelations) : null,
  };
}

function dedupeRelations(relations) {
  const seen = new Set();
  const out = [];
  for (const relation of relations) {
    const key = JSON.stringify([
      relation.relation,
      relation.kind ?? null,
      relation.name ?? relation.slot ?? null,
      relation.reason ?? null,
      relation.exposure ?? null,
      relation.item ?? null,
    ]);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(relation);
  }
  return out;
}

/**
 * Change each read port would see under `delta`.  Every *effective* frozen port
 * must report a change of zero (within tolerance); non-effective ports are
 * reported for information with `effective: false`.
 */
export function readPortsForUpdate(assets, delta, ports, constraints, tolerance = DEFAULT_TOLERANCE) {
  return describePorts(assets, ports, constraints, delta, tolerance).map((port) => ({
    name: port.name,
    kind: port.kind,
    profile: port.profile,
    effective: port.effective,
    change: port.change,
    preserved: port.preserved,
  }));
}

function describePorts(assets, ports, constraints, delta, tolerance) {
  const scale = Math.max(1, sparseNorm(delta));
  return portDirections(assets, ports, constraints).map((port) => {
    const change = changeFor(port.direction, delta);
    const threshold = tolerance * scale * Math.max(1, port.direction ? sparseNorm(port.direction) : 1);
    return {
      name: port.name,
      kind: port.kind,
      profile: port.profile,
      axes: port.axes,
      effective: port.effective,
      frozen: port.frozen,
      readable: port.readable,
      computable: change !== null,
      change,
      threshold,
      preserved: change === null ? null : Math.abs(change) <= threshold,
    };
  });
}

/**
 * Build the "continue with the previous legal read" payload.
 *
 * When nothing is applied the recommender keeps using whatever it already had.
 * Blocked relations get top billing; frozen ports from the snapshot are kept
 * too, because their values must not move either.  Non-frozen ports are
 * expected to be re-read normally.
 */
export function buildFallback(lastLegalRead, reason, relations) {
  const blockedPortNames = relations
    .filter((relation) => relation.kind === 'view' || relation.kind === 'mixture')
    .map((relation) => relation.name);
  const ports = Array.isArray(lastLegalRead?.ports) ? lastLegalRead.ports : Array.isArray(lastLegalRead) ? lastLegalRead : [];
  const frozenPortNames = ports.filter((port) => port?.frozen !== false).map((port) => port.name);
  return {
    reason,
    keep: [...new Set([...blockedPortNames, ...frozenPortNames])],
    blocked: [...new Set(blockedPortNames)],
    ports,
    detail: 'apply nothing; keep serving the last legal read for the listed ports',
  };
}

/**
 * Apply a proposal to a base parameter map.  A blocked proposal is never
 * applied; the caller gets the fallback snapshot instead.  A partial proposal
 * writes only the applicable components.
 */
export function commitUpdate(assets, base, proposal) {
  if (proposal.status === 'blocked') {
    return {applied: false, reason: proposal.blockReason, params: null, ports: null, fallback: proposal.fallback};
  }
  const written = proposal.written ?? proposal.delta;
  const params = new Map();
  for (const [slot, vector] of base) params.set(slot, Float64Array.from(vector));
  for (const [slot, delta] of written) {
    let target = params.get(slot);
    if (!target) {
      target = new Float64Array(assets.dimension);
      params.set(slot, target);
    }
    for (let i = 0; i < target.length; i += 1) target[i] += delta[i] ?? 0;
  }
  return {
    applied: true,
    partial: proposal.status === 'partial',
    reason: proposal.blockReason,
    params,
    ports: proposal.ports,
    fallback: proposal.fallback,
  };
}

/** A parameters map from physical slot vectors; handy as the base of a step. */
export function paramsFromSlots(assets, keys) {
  const params = new Map();
  for (const key of keys) params.set(key, assets.slotValue(key));
  return params;
}

/** Snapshot a serving pass so it can be used as `lastLegalRead`. */
export function snapshotReads(reading) {
  return {
    policy: reading.policy ?? null,
    profile: reading.profile ?? null,
    ports: (reading.ports ?? []).map((port) => ({...port})),
    readouts: (reading.readouts ?? []).map((readout) => ({
      id: readout.id,
      fields: readout.fields,
      status: readout.status,
      value: readout.value,
      zero: readout.zero,
    })),
  };
}

export {readPorts};
