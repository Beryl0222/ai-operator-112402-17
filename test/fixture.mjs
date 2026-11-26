// Shared fixture helpers for the operator tests.
//
// The synthetic document below is small enough that every expected value can be
// computed by hand, while keeping the shapes the real device samples use:
// aliased fields, a closed domain, pair and triple contracts, a read-only
// view, and a joint read port over two fields.

import {createAssets} from '../src/assets.mjs';

export const DIM = 6;

export function slotVector(key) {
  return Float64Array.from({length: DIM}, (_, i) => ((key * 7 + i * 3) % 11) / 10 + 0.05);
}

export function syntheticSource() {
  return {
    fields: 8,
    dimension: DIM,
    slots: 32,
    workBytes: 4096,
    slotFetchLimit: 40,
    batch: 4,
    aliases: [
      [0, 5, 20], // fields 0 and 5 share slot 20
      [3, 6, 21], // fields 3 and 6 share slot 21
    ],
    policy: [
      ['open', [0, 1, 2, 3, 4, 5, 6, 7]],
      ['tight', [1, 2, 5, 6, 7]],
    ],
    pairs: [
      [0, 1, [0, 1, 3]],
      [5, 2, [0, 2, 4]],
      [3, 4, [1, 2, 5]],
    ],
    triples: [
      [0, 1, 5, [0, 1, 4]],
      [1, 2, 5, [0, 2, 3]],
    ],
    views: [
      ['read-only', 20, [1, 0.5, -0.25]],
      ['trainable', 21, [0.5, 1, 0]],
    ],
    eligibleAxis: [0, 1, 2, 3, 4, 5],
    profiles: [
      ['rank4', 4, 1e-9],
      ['rank6', 6, 1e-10],
    ],
    protectedMixtures: [
      [
        'joint',
        'rank4',
        [
          [5, 1, [0, 1, 2]],
          [2, -0.5, [0, 1, 2]],
        ],
        [1, 2, 1],
      ],
      [
        'deep',
        'rank6',
        [
          [1, 1, [0, 1, 3]],
        ],
        [1, 1, 0],
      ],
    ],
  };
}

/**
 * Build the fixture.  Field slots are explicit so tests never depend on the
 * sine calibration: field 0 and 5 share slot 20; field 3 and 6 share slot 21;
 * the remaining fields own slots 1..7, each with weight 1.
 */
export function syntheticAssets(overrides = {}) {
  const own = {
    1: [1],
    2: [2],
    4: [3],
    7: [4],
  };
  const shared = {0: [20], 5: [20], 3: [21], 6: [21]};
  return createAssets(
    {...syntheticSource(), ...overrides.source},
    {
      fieldSlots: (id) => {
        if (overrides.fieldSlots) return overrides.fieldSlots(id);
        const slots = shared[id] ?? own[id] ?? [id];
        return slots.map((key, index) => ({key, weight: index === 0 ? 1 : 0.5}));
      },
      slotValue: overrides.slotValue ?? slotVector,
    },
  );
}

/** Reference embedding computed straight from the fixture definition. */
export function embeddingOf(assets, field) {
  const vector = new Float64Array(assets.dimension);
  for (const {key, weight} of assets.fieldSlots(field)) {
    const value = slotVector(key);
    for (let i = 0; i < vector.length; i += 1) vector[i] += weight * value[i];
  }
  return vector;
}

export function referenceTerm(assets, fields, axes) {
  const embeddings = fields.map((field) => embeddingOf(assets, field));
  let total = 0;
  for (const axis of axes) {
    let product = 1;
    for (const vector of embeddings) product *= vector[axis];
    total += product;
  }
  return total;
}
