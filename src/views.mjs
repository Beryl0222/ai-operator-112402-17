// Read ports and the linear constraints they impose on a proposed update.
//
// A public view is a direction over one physical slot; a protected mixture is
// a joint view over the physical slots of several fields.  Both become sparse
// directions in slot space, because a shared alias means the participating
// fields literally write to the same coordinates: two domains can cancel
// inside a joint view, and zeroing one domain would throw away a personalized
// change the other domain is still entitled to make.
//
// A mixture may be declared at one tier and still matter for a commit at
// another.  When the same target was received across tiers, the caller passes
// every tier the target actually spans, and every applicable mixture of those
// tiers becomes part of the same system of linear equalities.  Equations that
// share a slot can be linearly dependent; projection orthonormalises them.
//
// "A lattice point beyond authority is unreadable": a port whose axes are not
// in the dataset's eligible lattice, or whose member domain is closed, is
// reported as a blocked relation instead of being approximated.

import {zeros} from './linear.mjs';

export function addAxis(direction, slot, axis, amount) {
  let vector = direction.get(slot);
  if (!vector) {
    vector = zeros(axis + 1);
    direction.set(slot, vector);
  } else if (vector.length <= axis) {
    const grown = zeros(axis + 1);
    grown.set(vector);
    vector = grown;
    direction.set(slot, vector);
  }
  vector[axis] += amount;
  return direction;
}

function normaliseDirection(direction, dimension) {
  for (const [slot, vector] of direction) {
    const dense = zeros(dimension);
    for (let i = 0; i < vector.length && i < dimension; i += 1) dense[i] = vector[i];
    direction.set(slot, dense);
  }
  return direction;
}

/**
 * Direction of a plain view: the 3-tuple of weights applies to axes [0,1,2],
 * and every undeclared axis has coefficient zero.  Returns null when the
 * declared weights reach past the physical dimension.
 */
export function viewDirection(assets, view) {
  if (view.weights.length > assets.dimension) return null;
  const vector = zeros(assets.dimension);
  for (let axis = 0; axis < view.weights.length; axis += 1) vector[axis] = view.weights[axis];
  return new Map([[view.key, vector]]);
}

/**
 * Direction of a protected mixture.
 *
 * Each member's embedding is first interpreted from its physical slots (slot
 * weight x member coefficient), then remapped by `mapping` to the read-port
 * position, and finally dotted with the port weights.  The result is the
 * sparse slot-space gradient of that scalar -- exactly the direction that must
 * be orthogonal to the update.
 */
export function mixtureDirection(assets, mixture) {
  const direction = new Map();
  const axes = [];
  for (const member of mixture.members) {
    const refs = assets.fieldSlots(member.field);
    if (!refs || !refs.length) return {direction: null, axes: null, reason: 'missing-field'};
    for (let j = 0; j < mixture.weights.length; j += 1) {
      const axis = member.mapping[j];
      axes.push(axis);
      for (const {key, weight} of refs) {
        addAxis(direction, key, axis, member.coefficient * weight * mixture.weights[j]);
      }
    }
  }
  return {direction: normaliseDirection(direction, assets.dimension), axes, reason: null};
}

/** Value of a read port at `base` (Map<slot, Float64Array> or a lookup fn). */
export function portValue(direction, base) {
  let total = 0;
  for (const [slot, weights] of direction) {
    const vector = typeof base === 'function' ? base(slot) : base.get(slot);
    if (!vector) return null;
    for (let axis = 0; axis < weights.length; axis += 1) {
      if (weights[axis] !== 0) total += weights[axis] * vector[axis];
    }
  }
  return total;
}

function axesReadable(assets, axes) {
  return axes.every((axis) => assets.eligibleAxis.has(axis));
}

function axesWithin(assets, axes, bound) {
  return axes.every((axis) => Number.isInteger(axis) && axis >= 0 && axis < bound && assets.eligibleAxis.has(axis));
}

/**
 * Build the frozen constraints for a commit at `profile`.
 *
 * A frozen port is readable at a commit tier when every axis it maps to is an
 * eligible lattice point *and* inside that tier's bound ("格点越权不可读").
 * That is deliberately about axes rather than about the tier the port was
 * declared at: a rank48 port whose axes all sit below 32 is still an observed
 * read at a rank32 commit, so a rank32 freeze can change a target that arrived
 * at rank48.  Such a port contributes its equation over the axes the commit
 * tier actually contains, which is what "以实际包含的全部合法轴为约束" asks
 * for.
 *
 * `profiles` lets the caller record every tier the target spans; the reachable
 * bound is still the commit tier, so a commit cannot become stricter than the
 * model it writes.
 */
export function frozenConstraints(assets, options = {}) {
  const {policy, profile = null, profiles = null, bound = null, includeMixtures = true} = options;
  const commitTier = profile ? assets.profile(profile) : null;
  if (profile && !commitTier) throw new Error(`unknown profile: ${profile}`);
  const reach = bound ?? commitTier?.bound ?? Infinity;
  const spans = profiles ?? (profile ? [profile] : []);
  const epsilon = Math.min(
    ...(spans.length ? spans : [profile])
      .map((name) => assets.profile(name)?.epsilon)
      .filter((value) => Number.isFinite(value)),
    Number.POSITIVE_INFINITY,
  );
  const policyFields = assets.policy(policy)?.fieldSet ?? new Set();

  const constraints = [];
  const blocked = [];
  const ports = [];

  for (const view of assets.views) {
    const direction = viewDirection(assets, view);
    const owners = assets.fieldsForSlot(view.key);
    const readable = owners.some((field) => policyFields.has(field));
    const port = {
      name: view.name,
      kind: 'view',
      slot: view.key,
      profile: null,
      axes: [...view.axes],
      direction,
      readable,
      frozen: !view.editable,
      fields: owners,
    };
    ports.push(port);
    if (view.editable) continue;
    if (!direction) {
      blocked.push({relation: 'read-port', name: view.name, kind: 'view', reason: 'out-of-dimension'});
      continue;
    }
    if (!readable) {
      blocked.push({relation: 'read-port', name: view.name, kind: 'view', reason: 'closed-slot'});
      continue;
    }
    if (!axesReadable(assets, view.axes)) {
      blocked.push({relation: 'read-port', name: view.name, kind: 'view', reason: 'beyond-lattice', axes: [...view.axes]});
      continue;
    }
    if (!axesWithin(assets, view.axes, reach)) {
      blocked.push({relation: 'read-port', name: view.name, kind: 'view', reason: 'beyond-tier', axes: [...view.axes]});
      continue;
    }
    constraints.push({
      name: view.name,
      kind: 'view',
      profile: null,
      epsilon: Number.isFinite(epsilon) ? epsilon : 1e-9,
      direction,
      axes: [...view.axes],
      fields: owners,
    });
  }

  if (includeMixtures) {
    for (const mixture of assets.protectedMixtures) {
      const mixtureTier = assets.profile(mixture.profile);
      if (!mixtureTier) continue;
      const {direction, axes, reason} = mixtureDirection(assets, mixture);
      const members = mixture.members.map((member) => member.field);
      const closed = members.filter((field) => !policyFields.has(field));
      ports.push({
        name: mixture.name,
        kind: 'mixture',
        profile: mixture.profile,
        fields: members,
        axes,
        direction,
        frozen: true,
        readable: closed.length === 0 && Boolean(direction),
      });

      if (reason) {
        blocked.push({relation: 'read-port', name: mixture.name, kind: 'mixture', reason});
        continue;
      }
      const inReach = axesReadable(assets, axes) && axesWithin(assets, axes, reach);
      if (!inReach) {
        // Not observable at the commit tier: reported, but it was never part of
        // this model's released region, so it cannot be violated either.
        blocked.push({
          relation: 'read-port',
          name: mixture.name,
          kind: 'mixture',
          reason: axesReadable(assets, axes) ? 'beyond-tier' : 'beyond-lattice',
          axes,
          profile: mixture.profile,
        });
        continue;
      }
      if (closed.length) {
        blocked.push({
          relation: 'read-port',
          name: mixture.name,
          kind: 'mixture',
          reason: 'domain-switch',
          closedFields: closed,
          profile: mixture.profile,
          axes,
        });
        continue;
      }
      constraints.push({
        name: mixture.name,
        kind: 'mixture',
        profile: mixture.profile,
        epsilon: mixtureTier.epsilon,
        direction,
        axes,
        fields: members,
      });
    }
  }

  return {constraints, ports, blocked, spans, epsilon, reach};
}

/** Forward-pass values for the read ports, subject to the policy. */
export function readPorts(assets, reader) {
  const results = [];
  for (const view of assets.views) {
    const owners = assets.fieldsForSlot(view.key);
    if (!owners.some((field) => reader.isAuthorized(field))) {
      results.push({name: view.name, kind: 'view', slot: view.key, available: false, reason: 'closed-slot', frozen: !view.editable});
      continue;
    }
    const value = reader.viewValue(view.key, view.weights, {purpose: 'inspect'});
    results.push({
      name: view.name,
      kind: 'view',
      slot: view.key,
      available: value.available,
      reason: value.available ? null : value.reason,
      value: value.available ? value.value : null,
      zero: value.available ? value.zero : null,
      frozen: !view.editable,
    });
  }
  for (const mixture of assets.protectedMixtures) {
    const closed = mixture.members.filter((member) => !reader.isAuthorized(member.field));
    if (closed.length) {
      results.push({
        name: mixture.name,
        kind: 'mixture',
        profile: mixture.profile,
        available: false,
        reason: 'domain-switch',
        closedFields: closed.map((member) => member.field),
        frozen: true,
      });
      continue;
    }
    const {direction} = mixtureDirection(assets, mixture);
    const value = direction ? portValue(direction, (slot) => readSlotValue(assets, reader, slot)) : null;
    results.push({
      name: mixture.name,
      kind: 'mixture',
      profile: mixture.profile,
      available: value !== null,
      reason: value === null ? 'unreadable' : null,
      value,
      zero: value === null ? null : value === 0,
      frozen: true,
    });
  }
  return results;
}

/**
 * Slot lookup for reporting a mixture value.  Reads go through the reader so
 * the access ledger stays complete; each slot is fetched at most once per
 * reader because the reader caches it.
 */
function readSlotValue(assets, reader, slot) {
  const endpoints = [];
  for (const alias of assets.aliases) if (alias.key === slot) endpoints.push(...alias.fields);
  const owners = endpoints.length ? endpoints : assets.fieldsForSlot(slot);
  const field = owners.find((candidate) => reader.isAuthorized(candidate));
  if (field === undefined) return undefined;
  const result = reader.slot(slot, {field, purpose: 'inspect'});
  return result.available ? result.value : undefined;
}
