// Public surface of the on-device sparse interaction operator.

export {createAssets, createRandom, loadAssetData, interactionKey} from './assets.mjs';
export {AccessLedger, SourceReader, auditCache, aliasEndpoints, slotEditable, slotId} from './access.mjs';
export {
  accumulateTrainingGradient,
  admissibleContributions,
  createTrainingItem,
  trainingBlockReason,
  trainingIdentity,
} from './training.mjs';
export {
  accumulateGradient,
  readTerm,
  termGradient,
  resolveTerm,
  readRegion,
  legalTermAxes,
} from './interactions.mjs';
export {
  frozenConstraints,
  mixtureDirection,
  portValue,
  readPorts,
  viewDirection,
} from './views.mjs';
export {
  claimTicket,
  commitAxes,
  createTickets,
} from './tickets.mjs';
export {
  domainOccupancy,
  sampleAxes,
  sampleRequest,
  sampleTerms,
  serve,
  serveAndLearn,
  serveRequest,
} from './serving.mjs';
export {
  commitUpdate,
  paramsFromSlots,
  proposeUpdate,
  snapshotReads,
} from './proposal.mjs';
export {
  componentCount,
  projectStep,
  sparseDot,
  sparseNorm,
  sparseSupport,
  validateConstraints,
} from './linear.mjs';
