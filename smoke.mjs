// End-to-end device smoke run: sample a request under the dataset's limits,
// serve it, learn the released terms, and hold the proposed step against the
// frozen read ports.  Prints a compact JSON summary; read-only.

import {loadAssetData} from './src/assets.mjs';
import {
  claimTicket,
  commitUpdate,
  createTickets,
  frozenConstraints,
  paramsFromSlots,
  proposeUpdate,
  sampleRequest,
  serveAndLearn,
  snapshotReads,
} from './src/index.mjs';

const assets = loadAssetData();
const plan = sampleRequest(assets, {seed: 11, policy: assets.latestPolicyName, profile: 'rank32'});
const lossWeights = Object.fromEntries(
  plan.terms.map((term, index) => [`${term.kind}:${term.fields.join(':')}`, 0.5 + index * 0.25]),
);

const learning = serveAndLearn(assets, {
  policy: plan.policy,
  profile: plan.profile,
  terms: plan.terms,
  requestedAxes: plan.requestedAxes,
  lossWeights,
});
const proposal = proposeUpdate(assets, {
  training: learning.training,
  exposure: plan.policy,
  profile: plan.profile,
  lastLegalRead: snapshotReads({policy: plan.policy, profile: plan.profile, ports: [], readouts: []}),
});

const base = paramsFromSlots(assets, [...learning.gradient.keys()]);
const committed = commitUpdate(assets, base, proposal);

const {constraints} = frozenConstraints(assets, {policy: plan.policy, profile: plan.profile});
const request = createTickets(assets, {policy: plan.policy, seed: 11, terms: plan.terms});
const low = claimTicket(assets, request, request.claim('rank32'), {lossWeights});
const high = claimTicket(assets, request, request.claim('rank48'), {lossWeights});

console.log(
  JSON.stringify(
    {
      plan: {
        policy: plan.policy,
        profile: plan.profile,
        terms: plan.terms.length,
        readAxes: plan.requestedAxes.length,
        intermediateBudget: plan.intermediateBudget,
        occupancy: plan.occupancy,
      },
      serve: {access: learning.access},
      learning: {
        admitted: learning.training.admitted,
        rejected: learning.training.rejected,
        blocked: learning.training.blocked,
        sharedSlots: learning.training.views
          .filter((view) => view.accumulated)
          .map((view) => ({slot: view.slot, objectives: view.objectives, fields: view.fields})),
      },
      proposal: {
        status: proposal.status,
        blockReason: proposal.blockReason,
        components: proposal.componentCount,
        rawComponents: proposal.rawComponentCount,
        rawNorm: Number(proposal.rawNorm.toFixed(9)),
        stepNorm: Number(proposal.stepNorm.toFixed(9)),
        removedNorm: Number(proposal.removedNorm.toFixed(9)),
        frozenConstraints: constraints.map((constraint) => ({
          name: constraint.name,
          kind: constraint.kind,
          axes: constraint.axes,
        })),
        frozenPorts: proposal.ports
          .filter((port) => port.effective)
          .map((port) => ({name: port.name, preserved: port.preserved})),
        blockedRelations: proposal.blockedRelations,
      },
      commit: {applied: committed.applied, partial: committed.partial ?? false},
      tickets: {
        sharedFetches: request.ledger.summary().fetches,
        low: {identity: low.training.itemResults[0]?.item, admitted: low.training.admitted},
        high: {identity: high.training.itemResults[0]?.item, admitted: high.training.admitted},
      },
    },
    null,
    2,
  ),
);
