// Dataset loading and the derived metadata the operator needs.
//
// Everything here is metadata: field -> (slot, weight) descriptions, the
// declared interaction contracts, profile bounds and policy memberships.  No
// interaction tensor and no field embedding is materialised at load time.

import {readFileSync} from 'node:fs';

export const DEFAULT_ASSET_URL = new URL('../inputs/field-assets.json', import.meta.url);

export function loadAssetData(url = DEFAULT_ASSET_URL) {
  const source = JSON.parse(readFileSync(url, 'utf8'));
  return createAssets(source);
}

export function interactionKey(fields) {
  return [...fields].sort((a, b) => a - b).join(':');
}

/**
 * Normalise the synthetic calibration document.
 *
 * `overrides.fieldSlots(id)` and `overrides.slotValue(key)` let tests pin
 * exact slot contents without depending on the sine calibration shipped with
 * the device samples.
 */
export function createAssets(source, overrides = {}) {
  const dimension = source.dimension;
  const fieldCount = source.fields;
  const slotCount = source.slots;

  const aliases = (source.aliases ?? []).map(([a, b, key]) => ({fields: [a, b], key}));
  const aliasKeyByField = new Map();
  for (const {fields: [a, b], key} of aliases) {
    if (!aliasKeyByField.has(a)) aliasKeyByField.set(a, key);
    if (!aliasKeyByField.has(b)) aliasKeyByField.set(b, key);
  }

  const fieldSlots =
    overrides.fieldSlots ??
    ((id) => [
      {key: aliasKeyByField.get(id) ?? (73 * id) % slotCount, weight: 0.5},
      {key: (97 * id + 7) % slotCount, weight: 0.375},
      {key: (113 * id + 13) % slotCount, weight: 0.125},
    ]);

  const slotValue =
    overrides.slotValue ??
    ((key) => Float64Array.from({length: dimension}, (_, i) => Math.sin(0.23 * (key + 13) * (i + 1)) * 0.25));

  const policies = new Map();
  for (const [name, fields] of source.policy ?? []) {
    policies.set(name, Object.freeze({name, fields: Object.freeze([...fields]), fieldSet: new Set(fields)}));
  }
  const policyNames = [...policies.keys()];

  const profiles = new Map();
  for (const [name, bound, epsilon] of source.profiles ?? []) {
    profiles.set(name, Object.freeze({name, bound, epsilon}));
  }

  const views = (source.views ?? []).map(([name, key, weights]) =>
    Object.freeze({
      name,
      key,
      weights: Object.freeze([...weights]),
      axes: Object.freeze(weights.map((_, index) => index)),
      editable: name !== 'read-only',
    }),
  );
  const viewByName = new Map(views.map((view) => [view.name, view]));
  const viewByKey = new Map(views.map((view) => [view.key, view]));

  const eligibleAxis = new Set(source.eligibleAxis ?? Array.from({length: dimension}, (_, i) => i));

  const slotRefsCache = new Map();
  const slotRefs = (field) => {
    if (!Number.isInteger(field) || field < 0 || field >= fieldCount) return null;
    if (!slotRefsCache.has(field)) {
      slotRefsCache.set(
        field,
        Object.freeze(fieldSlots(field).map(({key, weight}) => Object.freeze({key, weight}))),
      );
    }
    return slotRefsCache.get(field);
  };

  const slotToFields = new Map();
  for (let field = 0; field < fieldCount; field += 1) {
    for (const {key} of slotRefs(field)) {
      if (!slotToFields.has(key)) slotToFields.set(key, []);
      slotToFields.get(key).push(field);
    }
  }

  const pairs = (source.pairs ?? []).map(([a, b, axes]) =>
    Object.freeze({kind: 'pair', fields: Object.freeze([a, b]), axes: Object.freeze([...axes])}),
  );
  const triples = (source.triples ?? []).map(([a, b, c, axes]) =>
    Object.freeze({kind: 'triple', fields: Object.freeze([a, b, c]), axes: Object.freeze([...axes])}),
  );

  const contracts = new Map();
  for (const contract of [...pairs, ...triples]) contracts.set(interactionKey(contract.fields), contract);

  const protectedMixtures = (source.protectedMixtures ?? []).map(([name, profile, members, weights]) =>
    Object.freeze({
      name,
      profile,
      members: Object.freeze(
        members.map(([field, coefficient, mapping]) =>
          Object.freeze({field, coefficient, mapping: Object.freeze([...mapping])}),
        ),
      ),
      weights: Object.freeze([...weights]),
    }),
  );

  const assets = {
    source,
    fields: fieldCount,
    dimension,
    slots: slotCount,
    workBytes: source.workBytes,
    slotFetchLimit: source.slotFetchLimit,
    batch: source.batch,

    aliases: Object.freeze(aliases),
    policies,
    policyNames,
    profiles,
    views: Object.freeze(views),
    contracts,
    eligibleAxis,

    pairs,
    triples,
    protectedMixtures,

    fieldSlots: slotRefs,
    fieldsForSlot: (key) => [...(slotToFields.get(key) ?? [])],
    fieldsForSlotSet: (key) => new Set(slotToFields.get(key) ?? []),
    slotValue: (key) => slotValue(key),

    policy: (name) => policies.get(name) ?? null,
    profile: (name) => profiles.get(name) ?? null,
    view: (name) => viewByName.get(name) ?? null,
    viewForSlot: (key) => viewByKey.get(key) ?? null,
    contract(kind, fields) {
      const found = contracts.get(interactionKey(fields));
      return found && found.kind === kind ? found : null;
    },

    /** Axis ids a term may use under `profileName`, with the reason for drops. */
    legalAxes(axes, profileName, options = {}) {
      const profile = profiles.get(profileName);
      if (!profile) throw new Error(`unknown profile: ${profileName}`);
      const requested = options.requested ? new Set(options.requested) : null;
      const legal = [];
      const dropped = [];
      for (const axis of axes) {
        if (!Number.isInteger(axis) || axis < 0 || axis >= dimension) {
          dropped.push({axis, reason: 'out-of-range'});
        } else if (requested && !requested.has(axis)) {
          dropped.push({axis, reason: 'not-requested'});
        } else if (!eligibleAxis.has(axis)) {
          dropped.push({axis, reason: 'not-eligible'});
        } else if (axis >= profile.bound) {
          dropped.push({axis, reason: 'beyond-rank'});
        } else {
          legal.push(axis);
        }
      }
      return {legal, dropped};
    },

    /** True when `axis` is a readable lattice point for a profile tier. */
    axisAllowed(axis, profileName) {
      const profile = profiles.get(profileName);
      return Boolean(
        profile && Number.isInteger(axis) && axis >= 0 && axis < profile.bound && eligibleAxis.has(axis),
      );
    },

    latestPolicyName: policyNames[policyNames.length - 1],
  };

  return Object.freeze(assets);
}

/** Deterministic little PRNG so sampled requests are reproducible. */
export function createRandom(seed = 1) {
  let state = seed >>> 0 || 1;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  return {
    next,
    int(max) {
      return Math.floor(next() * max);
    },
    pick(list) {
      return list.length ? list[Math.floor(next() * list.length)] : undefined;
    },
    shuffle(list) {
      const out = [...list];
      for (let i = out.length - 1; i > 0; i -= 1) {
        const j = Math.floor(next() * (i + 1));
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    },
  };
}
