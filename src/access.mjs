// Authorization, source-slot accounting and the request-scoped reader.
//
// Two rules are kept deliberately separate:
//   * parameter sharing -- several fields may alias the same physical slot;
//   * authorization     -- which fields the current policy still exposes.
//
// A shared slot stays readable through a still-authorized alias, but a closed
// field never triggers a new slot fetch, not even for an identity it already
// has cached.  The ledger records both the calling field and the purpose of
// every logical slot call so the audit trail can be replayed later.

const DEFAULT_IDENTITY = {table: 0, version: 1};

export function slotIdentity(key, identity = DEFAULT_IDENTITY) {
  return {table: identity.table, version: identity.version, key};
}

export function slotId(identity) {
  return `${identity.table}:${identity.version}:${identity.key}`;
}

export class AccessLedger {
  constructor({slotFetchLimit = Infinity, workBytes = Infinity, dimension = 48, policy = null} = {}) {
    this.slotFetchLimit = slotFetchLimit;
    this.workBytes = workBytes;
    this.dimension = dimension;
    this.policy = policy;
    this.fetches = 0;
    this.calls = 0;
    this.cacheHits = 0;
    this.intermediateBytes = 0;
    this.records = [];
    this.observations = [];
    this.denials = [];
    this.callLimitHit = false;
    this.byteLimitHit = false;
  }

  get withinBudget() {
    return !this.callLimitHit && !this.byteLimitHit;
  }

  /**
   * Count one logical slot call.  Every call is observed with the domain it was
   * made for and the purpose it served; this is the device-side read log, and
   * the call count is what `slotFetchLimit` bounds.
   */
  call({key, field, purpose = 'read', cached = false, requirement = 'owner'}) {
    this.calls += 1;
    if (cached) this.cacheHits += 1;
    const observation = {index: this.calls, key, field, purpose, cached, requirement};
    this.observations.push(observation);
    if (this.calls > this.slotFetchLimit) {
      this.callLimitHit = true;
      this.deny({key, field, purpose, reason: 'slot-call-limit'});
      return {ok: false, reason: 'slot-call-limit', observation};
    }
    return {ok: true, observation};
  }

  deny({key, field, purpose, reason, owner = null}) {
    const denial = {key, field, purpose, reason, owner};
    this.denials.push(denial);
    return denial;
  }

  /** Account one unique source-slot fetch (cache misses only). */
  fetch({key, field, purpose, owner = null}) {
    this.fetches += 1;
    this.records.push({key, field, purpose, owner});
    return null;
  }

  /** Reserve working memory for intermediate coordinates, never a dense tensor. */
  reserveIntermediate(bytes, description = 'coordinates') {
    if (!Number.isFinite(bytes) || bytes < 0) return {ok: false, reason: 'invalid-size'};
    if (this.intermediateBytes + bytes > this.workBytes) {
      this.byteLimitHit = true;
      this.denials.push({reason: 'work-bytes', description, requested: bytes, used: this.intermediateBytes});
      return {ok: false, reason: 'work-bytes'};
    }
    this.intermediateBytes += bytes;
    return {ok: true};
  }

  summary() {
    return {
      calls: this.calls,
      fetches: this.fetches,
      cacheHits: this.cacheHits,
      slotFetchLimit: this.slotFetchLimit,
      intermediateBytes: this.intermediateBytes,
      workBytes: this.workBytes,
      withinBudget: this.withinBudget,
      overCallLimit: this.callLimitHit,
      overWorkBytes: this.byteLimitHit,
      denialCount: this.denials.length,
    };
  }
}

/** Fields named by the explicit alias table for a physical slot, if any. */
export function aliasEndpoints(assets, key) {
  const endpoints = new Set();
  for (const alias of assets.aliases) {
    if (alias.key !== key) continue;
    for (const field of alias.fields) endpoints.add(field);
  }
  return [...endpoints];
}

/**
 * A slot is editable when at least one explicit alias endpoint (or, for an
 * unaliased slot, any owner field) is still authorized.  A closed endpoint can
 * therefore keep holding the shared vector read-only without vetoing updates
 * that the remaining authorized domain is entitled to make.
 */
export function slotEditable(assets, key, policyFields) {
  const allowed = policyFields instanceof Set ? policyFields : new Set(policyFields);
  const endpoints = aliasEndpoints(assets, key);
  const owners = endpoints.length ? endpoints : assets.fieldsForSlot(key);
  return owners.some((field) => allowed.has(field));
}

/**
 * Request-scoped slot reader.
 *
 * `sharedCache` lets two release tickets of the same request reuse the source
 * reads (the "shared source read" rule) without re-fetching.  The cache stores
 * raw slot values only and is keyed by (table, version, key), so a value read at
 * one version identity can never be served for another.  Policy checks happen
 * on every call, cached or not.
 */
export class SourceReader {
  constructor(assets, {ledger, policy, purpose = 'read', sharedCache = null, profile = null} = {}) {
    this.assets = assets;
    this.ledger = ledger ?? new AccessLedger({dimension: assets.dimension});
    this.policy = policy ?? assets.latestPolicyName;
    this.policyFields = assets.policy(this.policy)?.fieldSet ?? new Set();
    this.purpose = purpose;
    this.profile = profile;
    this.cache = sharedCache ?? new Map();
    this.fieldCache = new Map();
    this.readPorts = [];
  }

  isAuthorized(field) {
    return this.policyFields.has(field);
  }

  fieldExists(field) {
    return Number.isInteger(field) && field >= 0 && field < this.assets.fields;
  }

  /**
   * Read one logical slot on behalf of `field`.
   *
   * `requireOwner` additionally demands that `field` is one of the physical
   * owners of the slot.  Serving embeddings pass it, because an authorized
   * field must not be able to pull another domain's private vector through a
   * slot number that happens to be in the parameter table.  Shared aliases and
   * read ports read without it, since the whole point is that an authorized
   * alias may observe the common vector.
   */
  slot(key, {field, purpose = this.purpose, owner = null, requireOwner = false} = {}) {
    if (!this.fieldExists(field)) {
      const reason = 'missing-field';
      this.ledger.deny({key, field, purpose, reason, owner});
      return {available: false, key, reason};
    }
    if (!this.isAuthorized(field)) {
      const reason = 'unauthorized';
      this.ledger.deny({key, field, purpose, reason, owner});
      return {available: false, key, reason};
    }
    if (requireOwner && !this.assets.fieldSlots(field)?.some((ref) => ref.key === key)) {
      const reason = 'not-an-owner';
      this.ledger.deny({key, field, purpose, reason, owner});
      return {available: false, key, reason};
    }
    const id = slotId(slotIdentity(key));
    const requirement = requireOwner ? 'owner' : 'shared';
    if (this.cache.has(id)) {
      const counted = this.ledger.call({key, field, purpose, cached: true, requirement});
      if (!counted.ok) return {available: false, key, reason: counted.reason};
      return {available: true, key, value: this.cache.get(id), cached: true};
    }
    this.ledger.fetch({key, field, purpose, owner});
    const counted = this.ledger.call({key, field, purpose, cached: false, requirement});
    if (!counted.ok) return {available: false, key, reason: counted.reason};
    const value = this.assets.slotValue(key);
    this.cache.set(id, value);
    return {available: true, key, value, cached: false};
  }

  /**
   * Read the first authorized owner's view of a logical slot.  This is how a
   * read port or a shared alias is observed without pretending the closed
   * endpoint is authorized.
   */
  slotForOwners(key, {purpose = this.purpose, label = null} = {}) {
    const owners = aliasEndpoints(this.assets, key);
    const candidates = owners.length ? owners : this.assets.fieldsForSlot(key);
    const field = candidates.find((candidate) => this.isAuthorized(candidate));
    if (field === undefined) {
      const reason = candidates.length ? 'unauthorized' : 'unowned-slot';
      this.ledger.deny({key, field: null, purpose, reason});
      return {available: false, key, reason, owners: candidates};
    }
    return this.slot(key, {field, purpose, owner: field});
  }

  /**
   * Full embedding of a field (sum of weighted physical slots).
   * Returns null with a reason when the field is missing or unauthorized.
   */
  embedding(field, {purpose = this.purpose} = {}) {
    if (!this.fieldExists(field)) return {available: false, reason: 'missing-field', field};
    if (!this.isAuthorized(field)) return {available: false, reason: 'unauthorized', field};
    if (this.fieldCache.has(field)) {
      const cached = this.fieldCache.get(field);
      return {available: true, field, vector: cached, cached: true};
    }
    const refs = this.assets.fieldSlots(field);
    if (!refs || !refs.length) return {available: false, reason: 'no-slots', field};
    const vector = new Float64Array(this.assets.dimension);
    for (const {key, weight} of refs) {
      const result = this.slot(key, {field, purpose, owner: field, requireOwner: true});
      if (!result.available) return {available: false, reason: result.reason, field, slot: key};
      for (let axis = 0; axis < vector.length; axis += 1) {
        vector[axis] += weight * result.value[axis];
      }
    }
    this.fieldCache.set(field, vector);
    return {available: true, field, vector, cached: false};
  }

  /** Read port value on `slot`: dot(weights, slot[0..weights.length-1]). */
  viewValue(key, weights, {purpose = this.purpose} = {}) {
    const result = this.slotForOwners(key, {purpose});
    if (!result.available) return {...result, value: null};
    let value = 0;
    for (let i = 0; i < weights.length; i += 1) {
      value += weights[i] * (result.value[i] ?? 0);
    }
    return {available: true, key, value, zero: value === 0, cached: result.cached};
  }
}

/**
 * Audit cache contents after a policy change.
 *
 * `held` lists closed-domain identities that may be inspected but must not be
 * re-read; `reachable` lists slots that an authorized alias may still read;
 * `aliasShared` pairs the closed and open endpoints that share a slot.
 */
export function auditCache(assets, cache, beforePolicy, afterPolicy) {
  const before = new Set(assets.policy(beforePolicy)?.fields ?? beforePolicy);
  const after = new Set(assets.policy(afterPolicy)?.fields ?? afterPolicy);
  const closed = [...before].filter((field) => !after.has(field));
  const closedSet = new Set(closed);
  const held = [];
  const reachable = [];
  const aliasShared = [];
  const seen = new Set();
  for (const id of cache.keys()) {
    const key = Number(id.split(':').at(-1));
    const endpoints = aliasEndpoints(assets, key);
    const owners = endpoints.length ? endpoints : assets.fieldsForSlot(key);
    if (owners.some((field) => closedSet.has(field))) held.push({id, key, owners});
    if (owners.some((field) => after.has(field))) reachable.push({id, key, owners});
    for (const alias of assets.aliases) {
      if (alias.key !== key) continue;
      const [a, b] = alias.fields;
      const closedField = closedSet.has(a) ? a : closedSet.has(b) ? b : null;
      const openField = after.has(a) ? a : after.has(b) ? b : null;
      if (closedField !== null && openField !== null && !seen.has(key)) {
        seen.add(key);
        aliasShared.push({key, closed: closedField, open: openField});
      }
    }
  }
  return {closed, held, reachable, aliasShared};
}
