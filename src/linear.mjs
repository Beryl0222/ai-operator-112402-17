// Sparse linear algebra for the on-device operator.
//
// A sparse vector is a Map<logicalSlotKey, Float64Array>.  Everything here
// works on the handful of physical slots a request actually touches; no
// fields x fields (or slots x slots) tensor is ever materialised.

export function zeros(dimension) {
  return new Float64Array(dimension);
}

export function vectorFromComponents(dimension, entries = []) {
  const vector = zeros(dimension);
  for (const [axis, value] of entries) {
    if (Number.isInteger(axis) && axis >= 0 && axis < dimension) vector[axis] += value;
  }
  return vector;
}

export function sparseClone(vector) {
  const out = new Map();
  for (const [key, value] of vector) out.set(key, Float64Array.from(value));
  return out;
}

export function sparseAxpy(dest, src, scale = 1) {
  for (const [key, value] of src) {
    let target = dest.get(key);
    if (!target) {
      target = zeros(value.length);
      dest.set(key, target);
    }
    const n = Math.min(target.length, value.length);
    for (let i = 0; i < n; i += 1) target[i] += scale * value[i];
  }
  return dest;
}

export function sparseDot(a, b) {
  let sum = 0;
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  for (const [key, value] of small) {
    const other = large.get(key);
    if (!other) continue;
    const n = Math.min(value.length, other.length);
    for (let i = 0; i < n; i += 1) sum += value[i] * other[i];
  }
  return sum;
}

export function sparseNormSq(vector) {
  let sum = 0;
  for (const value of vector.values()) {
    for (let i = 0; i < value.length; i += 1) sum += value[i] * value[i];
  }
  return sum;
}

export function sparseNorm(vector) {
  return Math.sqrt(sparseNormSq(vector));
}

/**
 * H-norm of a sparse step.  `hWeights` is either a per-axis array (same for
 * every slot) or a Map<slot, Float64Array>.  Training supplies these weights;
 * the projection itself stays Euclidean, as the serving contract requires.
 */
export function weightedNormSq(vector, hWeights) {
  let sum = 0;
  for (const [slot, value] of vector) {
    const weights = hWeights instanceof Map ? hWeights.get(slot) : hWeights;
    for (let i = 0; i < value.length; i += 1) {
      const weight = weights?.[i] ?? 1;
      sum += weight * value[i] * value[i];
    }
  }
  return sum;
}

export function weightedNorm(vector, hWeights) {
  return Math.sqrt(weightedNormSq(vector, hWeights));
}

export function sparseScale(vector, scale) {
  for (const value of vector.values()) {
    for (let i = 0; i < value.length; i += 1) value[i] *= scale;
  }
  return vector;
}

export function sparseAdd(dest, src) {
  return sparseAxpy(dest, src, 1);
}

/** Remove components at or below `tolerance`; keep any slot that still has one. */
export function pruned(vector, tolerance = 1e-15) {
  const out = new Map();
  for (const [key, value] of vector) {
    let any = false;
    for (let i = 0; i < value.length; i += 1) {
      if (Math.abs(value[i]) <= tolerance) value[i] = 0;
      else any = true;
    }
    if (any) out.set(key, value);
  }
  return out;
}

/** Applicable, non-zero scalar components grouped by logical slot. */
export function sparseSupport(vector, tolerance = 0) {
  const support = [];
  for (const [key, value] of [...vector.entries()].sort((a, b) => a[0] - b[0])) {
    const dims = [];
    for (let i = 0; i < value.length; i += 1) {
      if (Math.abs(value[i]) > tolerance) dims.push(i);
    }
    if (dims.length) support.push({slot: key, dims});
  }
  return support;
}

export function componentCount(vector, tolerance = 0) {
  let count = 0;
  for (const value of vector.values()) {
    for (let i = 0; i < value.length; i += 1) if (Math.abs(value[i]) > tolerance) count += 1;
  }
  return count;
}

/**
 * Euclidean projection of `raw` onto { delta : dot(direction_i, delta) = 0 }.
 *
 * Directions are orthonormalised first.  This matters in practice: several
 * protected read ports can share an alias slot, or a joint read port can be a
 * linear combination of the same fields, so the equalities are often linearly
 * dependent.  A naive per-constraint subtraction would double count them and
 * leave a residual; Gram-Schmidt drops the redundant directions instead.
 *
 * The projection is joint over every slot named by either the raw step or the
 * constraints.  Per-field (or per-slot) zeroing is deliberately not performed:
 * authorized fields may still need to move a shared slot in opposite
 * directions so their common read view stays fixed.
 */
export function projectStep(raw, constraints, options = {}) {
  const tolerance = options.tolerance ?? 1e-12;
  const basis = [];
  const dependent = [];
  const invalid = [];

  for (const constraint of constraints) {
    const direction = constraint.direction ?? constraint;
    const originalNorm = sparseNorm(direction);
    if (!Number.isFinite(originalNorm) || originalNorm === 0) {
      invalid.push(constraint.name ?? null);
      continue;
    }
    const orthogonal = sparseClone(direction);
    for (const q of basis) sparseAxpy(orthogonal, q.vector, -sparseDot(orthogonal, q.vector));
    const norm = sparseNorm(orthogonal);
    if (norm <= tolerance * Math.max(1, originalNorm)) {
      dependent.push(constraint.name ?? null);
      continue;
    }
    sparseScale(orthogonal, 1 / norm);
    basis.push({
      name: constraint.name ?? null,
      kind: constraint.kind ?? null,
      profile: constraint.profile ?? null,
      epsilon: constraint.epsilon ?? null,
      vector: orthogonal,
      directionNorm: originalNorm,
    });
  }

  const delta = sparseClone(raw);
  for (const q of basis) sparseAxpy(delta, q.vector, -sparseDot(delta, q.vector));

  const residuals = constraints.map((constraint) => {
    const direction = constraint.direction ?? constraint;
    return {
      name: constraint.name ?? null,
      kind: constraint.kind ?? null,
      profile: constraint.profile ?? null,
      epsilon: constraint.epsilon ?? null,
      directionNorm: sparseNorm(direction),
      dot: sparseDot(direction, delta),
    };
  });

  const rawNorm = sparseNorm(raw);
  const projectedNorm = sparseNorm(delta);
  return {
    delta: pruned(delta, options.pruneTolerance ?? 1e-15),
    basis,
    rank: basis.length,
    dependent,
    invalid,
    residuals,
    rawNorm,
    projectedNorm,
    removedNorm: Math.sqrt(Math.max(0, rawNorm * rawNorm - projectedNorm * projectedNorm)),
  };
}

/**
 * Check every applied frozen view against its tier tolerance.
 * A view is an inequality on the numerical residual, so a component that is
 * exactly zero is fine and partial zeroing is allowed.
 */
export function validateConstraints(delta, constraints, options = {}) {
  const scale = Math.max(1, sparseNorm(delta));
  return constraints.map((constraint) => {
    const direction = constraint.direction ?? constraint;
    const dot = sparseDot(direction, delta);
    const epsilon = constraint.epsilon ?? options.epsilon ?? 1e-9;
    const threshold = epsilon * scale * Math.max(1, sparseNorm(direction));
    return {
      name: constraint.name ?? null,
      kind: constraint.kind ?? null,
      profile: constraint.profile ?? null,
      dot,
      epsilon,
      threshold,
      satisfied: Math.abs(dot) <= threshold,
    };
  });
}

export function replaceVector(target, source) {
  target.clear();
  for (const [key, value] of source) target.set(key, Float64Array.from(value));
  return target;
}
