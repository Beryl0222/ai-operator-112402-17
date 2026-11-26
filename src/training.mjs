// Training-time gradient attribution for published, weighted, editable terms.
//
// A term may only backpropagate when all three hold:
//   * it was published (the release ticket that produced it exists);
//   * it carries a non-zero loss weight;
//   * every physical slot it writes still has an editable alias under the
//     policy in force when the gradient is committed.
//
// A vector reached by several views (aliases, a pair and a triple) gets an
// accumulated gradient, never the last writer's gradient.  Each contribution
// keeps the exposure and profile identity it was actually read under, because
// a queued v0 gradient does not become v1-licensable after a close event.

import {accumulateGradient, readTerm, resolveTerm, termGradient} from './interactions.mjs';
import {slotEditable} from './access.mjs';

/** Training objective identity: includes exposure *and* release tier. */
export function trainingIdentity({exposure, profile, kind, fields}) {
  return `${exposure}/${profile}/${kind}:${fields.join(':')}`;
}

export function createTrainingItem(assets, options) {
  const {
    id = null,
    kind,
    fields,
    profile,
    exposure,
    published = true,
    lossWeight = 1,
    upstream = 1,
    requestedAxes = null,
    purpose = 'loss',
  } = options;
  const expected = kind === 'pair' ? 2 : kind === 'triple' ? 3 : null;
  if (expected === null) throw new Error(`unknown term kind: ${kind}`);
  if (!Array.isArray(fields) || fields.length !== expected) {
    throw new Error(`${kind} needs ${expected} fields`);
  }
  const contract = assets.contract(kind, fields);
  if (!contract) throw new Error(`no published ${kind} contract for [${fields}]`);
  if (!assets.profile(profile)) throw new Error(`unknown profile: ${profile}`);
  if (!assets.policy(exposure)) throw new Error(`unknown policy: ${exposure}`);
  const identity = id ?? trainingIdentity({exposure, profile, kind, fields});
  return Object.freeze({
    id: identity,
    kind,
    fields: Object.freeze([...fields]),
    profile,
    exposure,
    published,
    lossWeight,
    upstream,
    requestedAxes: requestedAxes ? Object.freeze([...requestedAxes]) : null,
    purpose,
  });
}

/**
 * Why an item may not contribute at all.  Returns null when it may.
 *
 * Read authorization is checked per field; editability is checked per physical
 * slot.  A closed domain can keep holding a shared vector read-only, and the
 * open alias keeps that slot editable, so ownership alone never blocks a
 * legitimate update here.  Frozen read ports are enforced during projection.
 */
export function trainingBlockReason(assets, reader, item) {
  if (!item.published) return {reason: 'unpublished', detail: 'release ticket was never published'};
  if (!Number.isFinite(item.lossWeight) || item.lossWeight === 0) {
    return {reason: 'no-loss-weight', detail: 'term has no released loss weight'};
  }
  if (!Number.isFinite(item.upstream) || item.upstream === 0) {
    return {reason: 'zero-upstream', detail: 'dLoss/dvalue is zero'};
  }
  const {contract, reason} = resolveTerm(assets, item);
  if (!contract) return {reason: reason ?? 'no-contract', detail: 'no applicable interaction contract'};
  const policyFields = reader.policyFields;
  for (const field of contract.fields) {
    if (!reader.isAuthorized(field)) {
      return {reason: 'closed-domain', field, detail: 'field is not exposed by the current policy'};
    }
  }
  for (const field of contract.fields) {
    const refs = assets.fieldSlots(field);
    if (!refs || !refs.length) return {reason: 'missing-field', field, detail: 'field has no physical slots'};
    for (const {key} of refs) {
      if (!slotEditable(assets, key, policyFields)) {
        return {reason: 'alias-frozen', slot: key, field, detail: 'no editable alias remains'};
      }
    }
  }
  return null;
}

/**
 * Backward pass over `items`, accumulating into one sparse gradient.
 *
 * Returns per-item results, including the readout and its identity, so a later
 * commit can admit, restrict or reject each contribution by the tier and
 * exposure it was actually computed under.
 */
export function accumulateTrainingGradient(assets, reader, items, options = {}) {
  const profile = options.profile ?? reader.profile ?? reader.policy;
  const exposure = options.exposure ?? reader.policy;
  const requestedAxes = options.requestedAxes ?? null;
  const gradient = new Map();
  const itemResults = [];
  const contributions = [];
  const blocked = [];
  const bySlot = new Map();
  const seen = new Set();

  for (const item of items) {
    const identity =
      item.id ?? trainingIdentity({exposure: item.exposure ?? exposure, profile: item.profile ?? profile, kind: item.kind, fields: item.fields});
    if (seen.has(identity)) {
      blocked.push({item: identity, reason: 'duplicate-item', detail: 'same objective identity was submitted twice'});
      continue;
    }
    seen.add(identity);

    const itemProfile = item.profile ?? profile;
    const itemExposure = item.exposure ?? exposure;
    if (itemExposure !== reader.policy) {
      blocked.push({
        item: identity,
        reason: 'exposure-mismatch',
        detail: `item was read under ${itemExposure}, reader is at ${reader.policy}`,
      });
      itemResults.push({
        item: identity,
        identity,
        admitted: false,
        reason: 'exposure-mismatch',
        profile: itemProfile,
        exposure: itemExposure,
        gradient: new Map(),
      });
      continue;
    }

    const blockedReason = trainingBlockReason(assets, reader, item);
    if (blockedReason) {
      blocked.push({item: identity, ...blockedReason});
      itemResults.push({item: identity, identity, admitted: false, ...blockedReason, gradient: new Map()});
      continue;
    }

    const readout = readTerm(assets, reader, {...item, purpose: item.purpose ?? 'loss'}, {
      profile: itemProfile,
      requestedAxes: item.requestedAxes ?? requestedAxes,
    });
    if (!readout.applicable) {
      const reason = readout.reason ?? 'unavailable';
      blocked.push({item: identity, reason, detail: 'term is unavailable at its recorded profile'});
      itemResults.push({item: identity, identity, admitted: false, reason, gradient: new Map(), readout});
      continue;
    }
    if (!readout.axes.length) {
      blocked.push({item: identity, reason: 'no-legal-axes', detail: 'tier exposes no axis for this term'});
      itemResults.push({item: identity, identity, admitted: false, reason: 'no-legal-axes', gradient: new Map(), readout});
      continue;
    }

    const backward = termGradient(assets, readout, item.upstream, {
      lossWeight: item.lossWeight,
      source: {item: identity, profile: itemProfile, exposure: itemExposure},
    });
    accumulateGradient(gradient, backward.gradient);

    for (const contribution of backward.contributions) {
      contributions.push({...contribution, item: identity, profile: itemProfile, exposure: itemExposure});
      if (!bySlot.has(contribution.slot)) bySlot.set(contribution.slot, []);
      bySlot.get(contribution.slot).push({
        item: identity,
        field: contribution.field,
        weight: contribution.weight,
        profile: itemProfile,
        exposure: itemExposure,
      });
    }

    itemResults.push({
      item: identity,
      identity,
      admitted: true,
      kind: readout.kind,
      fields: readout.fields,
      profile: itemProfile,
      exposure: itemExposure,
      axes: readout.axes,
      value: readout.value,
      zero: readout.zero,
      lossWeight: item.lossWeight,
      upstream: item.upstream,
      readout,
      gradient: backward.gradient,
      contributionCount: backward.contributions.length,
    });
  }

  // A physical slot seen through several views (alias + pair + triple) shows
  // up once here with every contributing objective listed.
  const views = [...bySlot.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([slot, entries]) => {
      const objectives = [...new Set(entries.map((entry) => entry.item))];
      const fields = [...new Set(entries.map((entry) => entry.field))];
      return {
        slot,
        objectives,
        fields,
        contributionCount: entries.length,
        accumulated: objectives.length > 1 || entries.length > 1,
      };
    });

  return {
    gradient,
    itemResults,
    contributions,
    blocked,
    views,
    profile,
    exposure,
    admitted: itemResults.filter((result) => result.admitted).length,
    rejected: itemResults.filter((result) => !result.admitted).length,
  };
}

function restrictReadout(readout, axes) {
  const keep = new Set(axes);
  return {
    ...readout,
    axes: readout.axes.filter((axis) => keep.has(axis)),
    contributions: (readout.contributions ?? []).filter((entry) => keep.has(entry.axis)),
  };
}

/**
 * Admit the gradient of `training` for a commit at {exposure, profile}.
 *
 * * A stale exposure (a domain switch since the read) rejects the contribution:
 *   a queued read cannot be re-licensed.  The applicable remaining components
 *   are still projected and returned.
 * * Axes outside the commit tier are dropped ("cannot reuse the loss belonging
 *   to newly added axes"), while axes shared between tiers remain usable.  The
 *   gradient is recomputed from the stored readout so the restriction is exact,
 *   not an approximation of the higher-tier loss.
 */
export function admissibleContributions(assets, training, {exposure, profile}) {
  const commitProfile = assets.profile(profile);
  if (!commitProfile) throw new Error(`unknown profile: ${profile}`);
  const admitted = [];
  const rejected = [];
  const restricted = [];

  for (const result of training.itemResults) {
    if (!result.admitted) continue;
    if (result.exposure !== exposure) {
      rejected.push({
        item: result.item,
        reason: 'stale-exposure',
        exposure: result.exposure,
        profile: result.profile,
        detail: 'contribution was read under a different policy identity',
      });
      continue;
    }
    const sharedAxes = result.axes.filter((axis) => assets.axisAllowed(axis, profile));
    if (!sharedAxes.length) {
      rejected.push({
        item: result.item,
        reason: 'new-axis-loss',
        axes: result.axes,
        profile,
        detail: 'no axis of this contribution is within the commit tier',
      });
      continue;
    }
    if (sharedAxes.length === result.axes.length) {
      admitted.push({...result, appliedAxes: [...result.axes], droppedAxes: []});
      continue;
    }
    const backward = termGradient(assets, restrictReadout(result.readout, sharedAxes), result.upstream, {
      lossWeight: result.lossWeight,
      source: {item: result.item, profile: result.profile, exposure: result.exposure},
    });
    const droppedAxes = result.axes.filter((axis) => !sharedAxes.includes(axis));
    const entry = {
      ...result,
      gradient: backward.gradient,
      axes: sharedAxes,
      appliedAxes: sharedAxes,
      droppedAxes,
    };
    admitted.push(entry);
    restricted.push({item: result.item, keptAxes: sharedAxes, droppedAxes, profile});
  }

  const gradient = new Map();
  for (const result of admitted) accumulateGradient(gradient, result.gradient);
  return {gradient, admitted, rejected, restricted};
}
