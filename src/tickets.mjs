// Release tickets: one publish request, two independently claimable tiers.
//
// The request is published with two fit-axis upper bounds (e.g. rank32 and
// rank48).  Either ticket may be claimed on its own; claiming both shares the
// source reads of the same request but *not* the loss belonging to the newly
// added axes.  Objective identity therefore includes the ticket's tier, so a
// loss computed under rank32 cannot be filed as a rank48 objective, and a
// rank48 loss may only reuse the axes the commit tier actually exposes.
//
// Each ticket records the exposure it was actually read at.  A ticket claimed
// before a close event stays a v0 identity: the queued computation does not
// become v1-licensable, and a commit re-verifies it against the identity it
// really used.

import {AccessLedger, SourceReader, auditCache} from './access.mjs';
import {accumulateTrainingGradient, trainingIdentity} from './training.mjs';
import {sampleAxes, sampleTerms} from './serving.mjs';

/**
 * Publish one request with two independently claimable tiers.
 *
 * `tiers` defaults to the document's two declared profiles.  Both tickets read
 * the same source slots through a shared cache, but their objective identities
 * differ, so the higher tier's new-axis loss is never mistaken for the lower
 * tier's loss.
 */
export function createTickets(assets, options = {}) {
  const {
    policy = assets.latestPolicyName,
    tiers = defaultTiers(assets),
    seed = 1,
    limit = assets.batch,
    terms = null,
    requestedAxes = null,
    sharedCache = null,
    ledger = null,
  } = options;

  const selected = terms ?? sampleTerms(assets, {seed, policy, profile: tiers[0], limit});
  const cache = sharedCache ?? new Map();
  const sourceLedger = ledger ?? new AccessLedger({
    slotFetchLimit: assets.slotFetchLimit,
    workBytes: assets.workBytes,
    dimension: assets.dimension,
    policy,
  });

  const tickets = [];
  for (const tier of tiers) {
    const profile = assets.profile(tier);
    if (!profile) throw new Error(`unknown profile: ${tier}`);
    const axes = requestedAxes ?? sampleAxes(assets, tier, {seed});
    const reader = new SourceReader(assets, {
      ledger: sourceLedger,
      policy,
      purpose: 'loss',
      sharedCache: cache,
      profile: tier,
    });
    tickets.push({
      request: seed,
      policy,
      profile: tier,
      bound: profile.bound,
      axes: [...axes],
      reader,
      sharedCache: cache,
      ledger: sourceLedger,
      identity(candidate) {
        if (!candidate || !Array.isArray(candidate.fields)) return null;
        return trainingIdentity({
          exposure: policy,
          profile: tier,
          kind: candidate.kind,
          fields: candidate.fields,
        });
      },
    });
  }

  return {
    policy,
    seed,
    terms: selected,
    tickets,
    sharedCache: cache,
    ledger: sourceLedger,
    /** Claim one ticket by tier name. */
    claim(tier) {
      return tickets.find((ticket) => ticket.profile === tier) ?? null;
    },
    /**
     * Axes this ticket may actually learn on: the tier bound intersected with
     * the request's read region.  The other ticket's new axes are excluded.
     */
    learnableAxes(tier) {
      const profile = assets.profile(tier);
      if (!profile) return [];
      return sampleAxes(assets, tier, {seed}).filter((axis) => axis < profile.bound);
    },
    /** Cache identities held by the request, for the post-close audit. */
    auditCache(before, after) {
      return auditCache(assets, cache, before, after);
    },
  };
}

/**
 * Claim a ticket's training pass.
 *
 * `lossWeights` is keyed by the plain term id (`pair:a:b`); the item identity
 * includes the ticket's exposure and tier, so the same term claimed at two
 * tiers produces two distinct objectives rather than silently reusing one.
 */
export function claimTicket(assets, request, ticket, options = {}) {
  const {lossWeights = {}, published = null, requestedAxes = ticket.axes} = options;
  const items = [];
  for (const term of request.terms) {
    const id = `${term.kind}:${term.fields.join(':')}`;
    if (!(id in lossWeights)) continue;
    if (published !== null && !published(id, ticket.profile)) continue;
    items.push({
      id: ticket.identity({kind: term.kind, fields: term.fields}),
      termId: id,
      kind: term.kind,
      fields: term.fields,
      profile: ticket.profile,
      exposure: ticket.policy,
      published: true,
      lossWeight: lossWeights[id],
      upstream: term.upstream ?? 1,
      requestedAxes,
      purpose: 'loss',
    });
  }
  const training = accumulateTrainingGradient(assets, ticket.reader, items, {
    profile: ticket.profile,
    exposure: ticket.policy,
    requestedAxes,
  });
  return {
    ticket: ticket.profile,
    policy: ticket.policy,
    axes: [...ticket.axes],
    training,
    gradient: training.gradient,
    access: ticket.ledger.summary(),
  };
}

/**
 * Commit-tier axes for a multi-tier target: the union of every ticket's axes
 * actually inside the commit tier.  A rank32 commit keeps 0..31 and drops the
 * rank48-only lattice points (33, 41 here).
 */
export function commitAxes(assets, tiers, profile) {
  const commit = assets.profile(profile);
  if (!commit) throw new Error(`unknown profile: ${profile}`);
  const axes = new Set();
  for (const tier of tiers) {
    const declared = assets.profile(tier);
    if (!declared) continue;
    for (const axis of assets.eligibleAxis) {
      if (axis < Math.min(declared.bound, commit.bound)) axes.add(axis);
    }
  }
  return [...axes].sort((a, b) => a - b);
}

/** The document's two release tiers, lowest first. */
export function defaultTiers(assets) {
  return [...assets.profiles.keys()].sort((a, b) => assets.profiles.get(a).bound - assets.profiles.get(b).bound);
}
