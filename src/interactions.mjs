// Sparse cross-field interaction operator.
//
// A term is evaluated by gathering only the physical slots of the fields that
// the contract names.  Low-rank truncation selects the legal axes *before*
// anything is multiplied, so a closed or truncated domain is never touched by
// a "full tensor then mask" pass: there is no dense pair/triple tensor at all.
//
// Two zero-ish outcomes are kept apart on purpose:
//   * status "unavailable" -- a domain is missing or unauthorized, or the
//     contract itself is gone; there is no readout to speak of;
//   * status "ok", zero true -- the term was legitimately evaluated and the
//     low-rank filter (or cancellation) produced an exact zero.
// Consumers must not conflate the two when they decide what to serve.

import {zeros} from './linear.mjs';

export function termId(contract) {
  return `${contract.kind}:${contract.fields.join(':')}`;
}

export function resolveTerm(assets, term) {
  if (!term || !Array.isArray(term.fields)) return {contract: null, reason: 'malformed'};
  if (term.kind !== 'pair' && term.kind !== 'triple') return {contract: null, reason: 'unknown-kind'};
  const expected = term.kind === 'pair' ? 2 : 3;
  if (term.fields.length !== expected) return {contract: null, reason: 'bad-arity'};
  if (new Set(term.fields).size !== expected) return {contract: null, reason: 'duplicate-field'};
  const contract = assets.contract(term.kind, term.fields);
  if (!contract) return {contract: null, reason: 'no-contract'};
  return {contract, reason: null};
}

export function legalTermAxes(assets, contract, profileName, requestedAxes = null) {
  return assets.legalAxes(contract.axes, profileName, {requested: requestedAxes});
}

/**
 * Evaluate one contract.
 *
 * Reads happen field by field and only for authorized domains; a denied read
 * is reported, never silently replaced by zeros.  Contract axes are filtered
 * to the tier's legal rank and the caller's requested read region before any
 * arithmetic, so an unauthorized domain cannot even be "touched by a mask".
 */
export function readTerm(assets, reader, term, {profile, requestedAxes = null} = {}) {
  const id = term.id ?? `${term.kind}:${(term.fields ?? []).join(':')}`;
  const {contract, reason} = resolveTerm(assets, term);
  if (!contract) {
    return {
      id,
      kind: term.kind,
      fields: [...(term.fields ?? [])],
      applicable: false,
      status: 'unavailable',
      reason,
      value: null,
      zero: null,
      axes: [],
      unrequested: [],
      dropped: [],
      embeddings: null,
    };
  }

  // Authorize and existence-check every field *before* the first slot fetch.
  // A "full tensor then mask" pass would read the closed domain first and only
  // discard it afterwards; here neither a missing nor an unauthorized domain
  // causes any slot access at all.
  const missingFields = [];
  for (const field of contract.fields) {
    if (!reader.fieldExists(field)) missingFields.push({field, reason: 'missing-field'});
    else if (!reader.isAuthorized(field)) missingFields.push({field, reason: 'unauthorized'});
    else {
      const refs = assets.fieldSlots(field);
      if (!refs || !refs.length) missingFields.push({field, reason: 'no-slots'});
    }
  }
  if (missingFields.length) {
    const onlyUnauthorized = missingFields.every((entry) => entry.reason === 'unauthorized');
    return {
      id,
      kind: contract.kind,
      fields: [...contract.fields],
      applicable: false,
      status: 'unavailable',
      reason: onlyUnauthorized ? 'unauthorized' : 'missing-field',
      missingFields,
      value: null,
      zero: null,
      axes: [],
      unrequested: [],
      dropped: [],
      embeddings: null,
    };
  }

  // Truncate to the tier's legal axes and the requested read region before any
  // multiplication.  Nothing outside this list is ever touched.
  const {legal, dropped} = legalTermAxes(assets, contract, profile, requestedAxes);
  const requestedSet = requestedAxes ? new Set(requestedAxes) : null;
  const unrequested = contract.axes.filter((axis) => requestedSet && !requestedSet.has(axis));
  if (!legal.length) {
    // The term is available and really was evaluated; the read region simply
    // contains none of its axes.  That is a low-rank zero, not missing data.
    const reason = dropped.length ? 'no-legal-axes' : 'empty-axes';
    return {
      id,
      kind: contract.kind,
      fields: [...contract.fields],
      applicable: true,
      status: 'ok',
      reason,
      missingFields: [],
      value: 0,
      zero: true,
      zeroKind: 'low-rank',
      axes: [],
      unrequested,
      dropped,
      contributions: [],
      embeddings: null,
    };
  }

  const embeddings = [];
  for (const field of contract.fields) {
    const result = reader.embedding(field, {purpose: term.purpose ?? 'score'});
    if (!result.available) {
      return {
        id,
        kind: contract.kind,
        fields: [...contract.fields],
        applicable: false,
        status: 'unavailable',
        reason: result.reason,
        missingFields: [{field, reason: result.reason}],
        value: null,
        zero: null,
        axes: legal,
        unrequested,
        dropped,
        embeddings: null,
      };
    }
    embeddings.push(result);
  }

  const contributions = [];
  let value = 0;
  for (const axis of legal) {
    let product = 1;
    for (const entry of embeddings) product *= entry.vector[axis];
    contributions.push({axis, product});
    value += product;
  }

  return {
    id,
    kind: contract.kind,
    fields: [...contract.fields],
    applicable: true,
    status: 'ok',
    reason: value === 0 ? 'zero-value' : null,
    missingFields: [],
    value,
    // true means the term was actually evaluated and the low-rank sum (or a
    // legitimate cancellation) is exactly zero; it does not mean "missing".
    zero: value === 0,
    zeroKind: value === 0 ? 'cancellation-or-zero-embedding' : null,
    axes: legal,
    unrequested,
    dropped,
    contributions,
    embeddings: embeddings.map((entry) => entry.vector),
  };
}

/**
 * Backward pass for one already-read term.
 *
 * `upstream` is dLoss/d(value); `upstream * lossWeight` is spread over the
 * legal axes only.  Gradients are returned per *physical slot*, because that
 * is where shared aliases live and where accumulation between views happens.
 */
export function termGradient(assets, termReadout, upstream, {lossWeight = 1, source = null} = {}) {
  if (!termReadout.applicable || termReadout.status !== 'ok') {
    return {gradient: new Map(), contributions: [], blocked: []};
  }
  const gradient = new Map();
  const contributions = [];
  const factors = termReadout.embeddings;
  if (!factors || !factors.length) {
    return {gradient, contributions, blocked: [], scale: 0};
  }
  const scale = upstream * lossWeight;
  if (!Number.isFinite(scale) || scale === 0) {
    return {gradient, contributions, blocked: [], scale: 0};
  }

  const fields = termReadout.fields;
  for (let f = 0; f < fields.length; f += 1) {
    const field = fields[f];
    const embGrad = zeros(assets.dimension);
    for (const axis of termReadout.axes) {
      let product = 1;
      for (let other = 0; other < factors.length; other += 1) {
        if (other !== f) product *= factors[other][axis];
      }
      embGrad[axis] = scale * product;
    }
    for (const {key, weight} of assets.fieldSlots(field)) {
      let slotGrad = gradient.get(key);
      if (!slotGrad) {
        slotGrad = zeros(assets.dimension);
        gradient.set(key, slotGrad);
      }
      for (let axis = 0; axis < assets.dimension; axis += 1) slotGrad[axis] += weight * embGrad[axis];
      contributions.push({
        slot: key,
        field,
        weight,
        axes: termReadout.axes,
        source,
        lossWeight,
        upstream,
      });
    }
  }
  return {gradient, contributions, blocked: [], scale};
}

/**
 * Accumulate term gradients into a caller-owned gradient map.
 * A field can observe the same vector through several views (aliases, pair and
 * triple contracts); contributions are additive rather than overwritten.
 */
export function accumulateGradient(target, gradient) {
  for (const [key, value] of gradient) {
    let existing = target.get(key);
    if (!existing) {
      existing = zeros(value.length);
      target.set(key, existing);
    }
    for (let i = 0; i < value.length; i += 1) existing[i] += value[i];
  }
  return target;
}

/**
 * A request's declared read region.  Defaults to every eligible axis that the
 * tier permits; callers can narrow it (a device may only expose a subset).
 */
export function readRegion(assets, profileName, requestedAxes = null) {
  const profile = assets.profile(profileName);
  if (!profile) throw new Error(`unknown profile: ${profileName}`);
  if (requestedAxes) {
    return [...new Set(requestedAxes)].filter((axis) => assets.axisAllowed(axis, profileName));
  }
  const axes = [];
  for (let axis = 0; axis < profile.bound; axis += 1) {
    if (assets.eligibleAxis.has(axis)) axes.push(axis);
  }
  return axes;
}
