import { FEATURE_NAMES } from '../config.ts';
import { mulberry32 } from './preprocessing.ts';
import {
  DecisionTreeModel,
  LogisticRegressionModel,
  TreeNode,
  XGBoostHyperparams,
  XGBoostModel,
} from './types.ts';

function sigmoid(z: number): number {
  if (z > 30) return 1 - 1e-13;
  if (z < -30) return 1e-13;
  return 1 / (1 + Math.exp(-z));
}

function logit(p: number): number {
  const clamped = Math.min(1 - 1e-7, Math.max(1e-7, p));
  return Math.log(clamped / (1 - clamped));
}

export function computePrAucFast(yTrue: Uint8Array, probs: Float64Array): number {
  const n = yTrue.length;
  const order = new Array<number>(n);
  let totalPos = 0;
  for (let i = 0; i < n; i++) {
    order[i] = i;
    if (yTrue[i] === 1) totalPos++;
  }
  if (totalPos === 0) return 0;

  order.sort((a, b) => probs[b] - probs[a]);

  let tp = 0;
  let fp = 0;
  let prevRecall = 0;
  let prevPrecision = 1;
  let auc = 0;

  for (let i = 0; i < n; i++) {
    const idx = order[i];
    if (yTrue[idx] === 1) tp++;
    else fp++;

    const recall = tp / totalPos;
    const precision = tp / (tp + fp);
    auc += (recall - prevRecall) * ((precision + prevPrecision) / 2);
    prevRecall = recall;
    prevPrecision = precision;
  }
  return auc;
}

export function computeF1AtThreshold(yTrue: Uint8Array, probs: Float64Array, threshold: number): number {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const pred = probs[i] >= threshold ? 1 : 0;
    if (pred === 1 && yTrue[i] === 1) tp++;
    else if (pred === 1 && yTrue[i] === 0) fp++;
    else if (pred === 0 && yTrue[i] === 1) fn++;
  }
  const denom = 2 * tp + fp + fn;
  return denom === 0 ? 0 : (2 * tp) / denom;
}

/**
 * Downsamples majority class (Class=0) ONLY during expensive tree CV/fitting if > maxNeg,
 * preserving 100% of minority fraud instances so CPU/RAM constraints (8 GB RAM) are respected.
 */
function balanceMajorityForFit(
  x: Float64Array[],
  y: Uint8Array,
  maxNeg: number,
  seed: number
): { x: Float64Array[]; y: Uint8Array } {
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1) posIdx.push(i);
    else negIdx.push(i);
  }
  if (negIdx.length <= maxNeg) {
    return { x, y };
  }
  const rand = mulberry32(seed);
  const sampledNeg: number[] = [];
  for (let i = 0; i < negIdx.length; i++) {
    if (sampledNeg.length < maxNeg) {
      sampledNeg.push(negIdx[i]);
    } else {
      const j = Math.floor(rand() * (i + 1));
      if (j < maxNeg) sampledNeg[j] = negIdx[i];
    }
  }
  const combined = posIdx.concat(sampledNeg);
  for (let i = combined.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = combined[i];
    combined[i] = combined[j];
    combined[j] = t;
  }
  const outX = new Array<Float64Array>(combined.length);
  const outY = new Uint8Array(combined.length);
  for (let i = 0; i < combined.length; i++) {
    outX[i] = x[combined[i]];
    outY[i] = y[combined[i]];
  }
  return { x: outX, y: outY };
}

// ---------------------------------------------------------------------------
// 1. LOGISTIC REGRESSION (Class-Weighted + L2 + Adam Optimizer)
// ---------------------------------------------------------------------------

export function trainLogisticRegression(
  trainX: Float64Array[],
  trainY: Uint8Array,
  epochs = 120,
  l2Lambda = 0.002,
  useClassWeights = true
): LogisticRegressionModel {
  const t0 = performance.now();
  const fitData = balanceMajorityForFit(trainX, trainY, 24000, 101);
  const x = fitData.x;
  const y = fitData.y;
  const n = x.length;
  const d = FEATURE_NAMES.length;

  let posCount = 0;
  for (let i = 0; i < n; i++) if (y[i] === 1) posCount++;
  const negCount = n - posCount;
  // Balanced positive class weight capped for calibrated probabilities
  const posWeight = useClassWeights
    ? Math.min(25, Math.max(1, Math.sqrt(negCount / Math.max(1, posCount))))
    : 1;

  const weights = new Float64Array(d);
  let bias = logit(posCount / Math.max(1, n));

  const mW = new Float64Array(d);
  const vW = new Float64Array(d);
  let mB = 0;
  let vB = 0;
  const lr = 0.04;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const gradW = new Float64Array(d);
    let gradB = 0;
    let totalWeight = 0;

    for (let i = 0; i < n; i++) {
      const row = x[i];
      let z = bias;
      for (let j = 0; j < d; j++) z += weights[j] * row[j];
      const p = sigmoid(z);
      const w = y[i] === 1 ? posWeight : 1;
      const err = (p - y[i]) * w;
      gradB += err;
      for (let j = 0; j < d; j++) gradW[j] += err * row[j];
      totalWeight += w;
    }

    const invW = 1 / Math.max(1, totalWeight);
    const b1Pow = 1 - Math.pow(beta1, epoch);
    const b2Pow = 1 - Math.pow(beta2, epoch);

    const gB = gradB * invW;
    mB = beta1 * mB + (1 - beta1) * gB;
    vB = beta2 * vB + (1 - beta2) * gB * gB;
    bias -= lr * (mB / b1Pow) / (Math.sqrt(vB / b2Pow) + eps);

    for (let j = 0; j < d; j++) {
      const g = gradW[j] * invW + l2Lambda * weights[j];
      mW[j] = beta1 * mW[j] + (1 - beta1) * g;
      vW[j] = beta2 * vW[j] + (1 - beta2) * g * g;
      weights[j] -= lr * (mW[j] / b1Pow) / (Math.sqrt(vW[j] / b2Pow) + eps);
    }
  }

  return {
    modelId: 'logistic_regression',
    weights: Array.from(weights),
    bias,
    posWeight: Number(posWeight.toFixed(4)),
    l2Lambda,
    epochs,
    trainingTimeMs: Number((performance.now() - t0).toFixed(2)),
  };
}

export function predictLogisticRegression(
  model: LogisticRegressionModel,
  x: Float64Array[]
): Float64Array {
  const out = new Float64Array(x.length);
  const w = model.weights;
  const d = w.length;
  for (let i = 0; i < x.length; i++) {
    const row = x[i];
    let z = model.bias;
    for (let j = 0; j < d; j++) z += w[j] * row[j];
    out[i] = sigmoid(z);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 2. DECISION TREE CLASSIFIER (Class-Weighted CART with Pruning)
// ---------------------------------------------------------------------------

function buildCartNode(
  x: Float64Array[],
  y: Uint8Array,
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  posWeight: number,
  importances: Float64Array
): TreeNode {
  let posW = 0;
  let negW = 0;
  let posRaw = 0;
  for (const idx of indices) {
    if (y[idx] === 1) {
      posW += posWeight;
      posRaw++;
    } else {
      negW += 1;
    }
  }
  const totalW = posW + negW;
  const prob = totalW > 0 ? posW / totalW : 0;

  if (
    depth >= maxDepth ||
    indices.length < minSamplesSplit ||
    posRaw === 0 ||
    posRaw === indices.length
  ) {
    return { isLeaf: true, probability: prob, samples: indices.length };
  }

  const pParent = posW / totalW;
  const parentGini = 1 - pParent * pParent - (1 - pParent) * (1 - pParent);

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  const d = FEATURE_NAMES.length;
  const numBins = 20;

  for (let f = 0; f < d; f++) {
    const vals = new Array<number>(indices.length);
    for (let i = 0; i < indices.length; i++) vals[i] = x[indices[i]][f];
    vals.sort((a, b) => a - b);

    const minV = vals[0];
    const maxV = vals[vals.length - 1];
    if (maxV - minV < 1e-9) continue;

    let prevThresh = NaN;
    for (let b = 1; b < numBins; b++) {
      const thresh = vals[Math.floor((b / numBins) * vals.length)];
      if (thresh === prevThresh) continue;
      prevThresh = thresh;

      let lPosW = 0;
      let lNegW = 0;
      let lCount = 0;
      for (const idx of indices) {
        if (x[idx][f] <= thresh) {
          lCount++;
          if (y[idx] === 1) lPosW += posWeight;
          else lNegW += 1;
        }
      }
      const rCount = indices.length - lCount;
      if (lCount < minSamplesLeaf || rCount < minSamplesLeaf) continue;

      const lTotal = lPosW + lNegW;
      const rPosW = posW - lPosW;
      const rNegW = negW - lNegW;
      const rTotal = rPosW + rNegW;
      if (lTotal <= 0 || rTotal <= 0) continue;

      const pl = lPosW / lTotal;
      const pr = rPosW / rTotal;
      const giniL = 1 - pl * pl - (1 - pl) * (1 - pl);
      const giniR = 1 - pr * pr - (1 - pr) * (1 - pr);
      const weightedGini = (lTotal / totalW) * giniL + (rTotal / totalW) * giniR;
      const gain = parentGini - weightedGini;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = thresh;
      }
    }
  }

  if (bestFeature === -1 || bestGain <= 1e-6) {
    return { isLeaf: true, probability: prob, samples: indices.length };
  }

  importances[bestFeature] += bestGain * totalW;

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const idx of indices) {
    if (x[idx][bestFeature] <= bestThreshold) leftIdx.push(idx);
    else rightIdx.push(idx);
  }

  return {
    isLeaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    samples: indices.length,
    left: buildCartNode(
      x, y, leftIdx, depth + 1, maxDepth, minSamplesSplit, minSamplesLeaf, posWeight, importances
    ),
    right: buildCartNode(
      x, y, rightIdx, depth + 1, maxDepth, minSamplesSplit, minSamplesLeaf, posWeight, importances
    ),
  };
}

export function trainDecisionTree(
  trainX: Float64Array[],
  trainY: Uint8Array,
  maxDepth = 5,
  minSamplesLeaf = 12,
  useClassWeights = true
): DecisionTreeModel {
  const t0 = performance.now();
  const fitData = balanceMajorityForFit(trainX, trainY, 18000, 202);
  const x = fitData.x;
  const y = fitData.y;

  let posCount = 0;
  for (let i = 0; i < y.length; i++) if (y[i] === 1) posCount++;
  const negCount = y.length - posCount;
  const posWeight = useClassWeights
    ? Math.min(20, Math.max(1, Math.sqrt(negCount / Math.max(1, posCount))))
    : 1;

  const indices = Array.from({ length: x.length }, (_, i) => i);
  const importances = new Float64Array(FEATURE_NAMES.length);

  const root = buildCartNode(
    x, y, indices, 0, maxDepth, minSamplesLeaf * 2, minSamplesLeaf, posWeight, importances
  );

  let impSum = 0;
  for (let j = 0; j < importances.length; j++) impSum += importances[j];
  const normImportances = Array.from(importances).map((v) => (impSum > 0 ? v / impSum : 0));

  return {
    modelId: 'decision_tree',
    root,
    maxDepth,
    minSamplesLeaf,
    posWeight: Number(posWeight.toFixed(4)),
    featureImportances: normImportances,
    trainingTimeMs: Number((performance.now() - t0).toFixed(2)),
  };
}

function traverseProbability(node: TreeNode, row: Float64Array): number {
  let curr = node;
  while (!curr.isLeaf && curr.featureIndex !== undefined && curr.threshold !== undefined) {
    curr = row[curr.featureIndex] <= curr.threshold ? curr.left! : curr.right!;
  }
  return curr.probability ?? 0;
}

export function predictDecisionTree(model: DecisionTreeModel, x: Float64Array[]): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = traverseProbability(model.root, x[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 3. TUNED XGBOOST (Regularized 2nd-Order GBDT + Stratified CV + Early Stopping)
// ---------------------------------------------------------------------------

function buildXGBTree(
  x: Float64Array[],
  grads: Float64Array,
  hessians: Float64Array,
  indices: number[],
  depth: number,
  params: XGBoostHyperparams,
  colSubset: number[],
  importances: Float64Array
): TreeNode {
  let sumG = 0;
  let sumH = 0;
  for (const idx of indices) {
    sumG += grads[idx];
    sumH += hessians[idx];
  }

  const leafVal = -sumG / (sumH + params.regLambda);

  if (depth >= params.maxDepth || indices.length < 10 || sumH < params.minChildWeight * 2) {
    return { isLeaf: true, leafValue: leafVal, samples: indices.length };
  }

  const parentScore = (sumG * sumG) / (sumH + params.regLambda);
  const gamma = 0.05;
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  const numBins = 16;

  for (const f of colSubset) {
    const vals = new Array<number>(indices.length);
    for (let i = 0; i < indices.length; i++) vals[i] = x[indices[i]][f];
    vals.sort((a, b) => a - b);

    if (vals[vals.length - 1] - vals[0] < 1e-9) continue;

    let prevThresh = NaN;
    for (let b = 1; b < numBins; b++) {
      const thresh = vals[Math.floor((b / numBins) * vals.length)];
      if (thresh === prevThresh) continue;
      prevThresh = thresh;

      let lG = 0;
      let lH = 0;
      let lCount = 0;
      for (const idx of indices) {
        if (x[idx][f] <= thresh) {
          lG += grads[idx];
          lH += hessians[idx];
          lCount++;
        }
      }
      const rCount = indices.length - lCount;
      const rG = sumG - lG;
      const rH = sumH - lH;

      if (
        lCount < 5 ||
        rCount < 5 ||
        lH < params.minChildWeight ||
        rH < params.minChildWeight
      ) {
        continue;
      }

      const gain =
        0.5 *
          ((lG * lG) / (lH + params.regLambda) +
            (rG * rG) / (rH + params.regLambda) -
            parentScore) -
        gamma;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = thresh;
      }
    }
  }

  if (bestFeature === -1 || bestGain <= 0) {
    return { isLeaf: true, leafValue: leafVal, samples: indices.length };
  }

  importances[bestFeature] += bestGain;

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const idx of indices) {
    if (x[idx][bestFeature] <= bestThreshold) leftIdx.push(idx);
    else rightIdx.push(idx);
  }

  return {
    isLeaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    samples: indices.length,
    left: buildXGBTree(x, grads, hessians, leftIdx, depth + 1, params, colSubset, importances),
    right: buildXGBTree(x, grads, hessians, rightIdx, depth + 1, params, colSubset, importances),
  };
}

function traverseLeafValue(node: TreeNode, row: Float64Array): number {
  let curr = node;
  while (!curr.isLeaf && curr.featureIndex !== undefined && curr.threshold !== undefined) {
    curr = row[curr.featureIndex] <= curr.threshold ? curr.left! : curr.right!;
  }
  return curr.leafValue ?? 0;
}

function fitXGBoostCore(
  trainX: Float64Array[],
  trainY: Uint8Array,
  valX: Float64Array[] | null,
  valY: Uint8Array | null,
  params: XGBoostHyperparams,
  seed: number
): {
  baseScore: number;
  trees: TreeNode[];
  featureImportances: number[];
  earlyStoppedRound: number;
} {
  const rand = mulberry32(seed);
  const n = trainX.length;
  const d = FEATURE_NAMES.length;

  let posCount = 0;
  for (let i = 0; i < n; i++) if (trainY[i] === 1) posCount++;
  const baseScore = logit(Math.max(0.005, Math.min(0.5, posCount / Math.max(1, n))));

  const rawPreds = new Float64Array(n).fill(baseScore);
  const grads = new Float64Array(n);
  const hessians = new Float64Array(n);
  const importances = new Float64Array(d);

  const valRawPreds = valX ? new Float64Array(valX.length).fill(baseScore) : null;
  const valProbs = valX ? new Float64Array(valX.length) : null;

  const trees: TreeNode[] = [];
  let bestValPrAuc = -1;
  let bestRound = params.nEstimators;
  let roundsWithoutImprovement = 0;
  const patience = 6;

  for (let round = 0; round < params.nEstimators; round++) {
    for (let i = 0; i < n; i++) {
      const p = sigmoid(rawPreds[i]);
      const w = trainY[i] === 1 ? params.scalePosWeight : 1;
      grads[i] = (p - trainY[i]) * w;
      hessians[i] = Math.max(1e-6, p * (1 - p) * w);
    }

    // Row subsample (stratified/always retain minority class with higher probability)
    const subIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (trainY[i] === 1 || rand() < params.subsample) {
        subIdx.push(i);
      }
    }

    // Column subsample
    const allCols = Array.from({ length: d }, (_, idx) => idx);
    for (let i = allCols.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = allCols[i];
      allCols[i] = allCols[j];
      allCols[j] = tmp;
    }
    const colCount = Math.max(6, Math.round(d * params.colsampleBytree));
    const colSubset = allCols.slice(0, colCount);

    const tree = buildXGBTree(trainX, grads, hessians, subIdx, 0, params, colSubset, importances);
    trees.push(tree);

    for (let i = 0; i < n; i++) {
      rawPreds[i] += params.learningRate * traverseLeafValue(tree, trainX[i]);
    }

    if (valX && valY && valRawPreds && valProbs) {
      for (let i = 0; i < valX.length; i++) {
        valRawPreds[i] += params.learningRate * traverseLeafValue(tree, valX[i]);
        valProbs[i] = sigmoid(valRawPreds[i]);
      }
      const valPrAuc = computePrAucFast(valY, valProbs);
      if (valPrAuc > bestValPrAuc + 1e-4) {
        bestValPrAuc = valPrAuc;
        bestRound = round + 1;
        roundsWithoutImprovement = 0;
      } else {
        roundsWithoutImprovement++;
        if (round >= 10 && roundsWithoutImprovement >= patience) {
          break;
        }
      }
    }
  }

  const keptTrees = valX ? trees.slice(0, bestRound) : trees;
  let impSum = 0;
  for (let j = 0; j < importances.length; j++) impSum += importances[j];
  const normImp = Array.from(importances).map((v) => (impSum > 0 ? v / impSum : 0));

  return {
    baseScore,
    trees: keptTrees,
    featureImportances: normImp,
    earlyStoppedRound: keptTrees.length,
  };
}

export function trainTunedXGBoost(
  trainX: Float64Array[],
  trainY: Uint8Array,
  valX: Float64Array[],
  valY: Uint8Array
): XGBoostModel {
  const t0 = performance.now();
  const fitData = balanceMajorityForFit(trainX, trainY, 14000, 303);
  const fx = fitData.x;
  const fy = fitData.y;

  let posCount = 0;
  for (let i = 0; i < fy.length; i++) if (fy[i] === 1) posCount++;
  const negCount = fy.length - posCount;
  const basePosWeight = Math.min(20, Math.max(1, Math.sqrt(negCount / Math.max(1, posCount))));

  // Stratified 3-fold CV candidate grid (controlled for fast CPU execution)
  const candidates: XGBoostHyperparams[] = [
    {
      nEstimators: 26,
      maxDepth: 3,
      learningRate: 0.14,
      subsample: 0.85,
      colsampleBytree: 0.75,
      minChildWeight: 2,
      regLambda: 1.5,
      scalePosWeight: basePosWeight,
    },
    {
      nEstimators: 32,
      maxDepth: 4,
      learningRate: 0.12,
      subsample: 0.80,
      colsampleBytree: 0.70,
      minChildWeight: 3,
      regLambda: 2.0,
      scalePosWeight: basePosWeight,
    },
    {
      nEstimators: 36,
      maxDepth: 3,
      learningRate: 0.10,
      subsample: 0.90,
      colsampleBytree: 0.80,
      minChildWeight: 1.5,
      regLambda: 1.0,
      scalePosWeight: Math.min(25, basePosWeight * 1.25),
    },
  ];

  // Build 3 stratified CV folds
  const folds: Array<{ tx: Float64Array[]; ty: Uint8Array; vx: Float64Array[]; vy: Uint8Array }> = [];
  const kFolds = 3;
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  for (let i = 0; i < fy.length; i++) {
    if (fy[i] === 1) posIdx.push(i);
    else negIdx.push(i);
  }

  for (let k = 0; k < kFolds; k++) {
    const tx: Float64Array[] = [];
    const tyList: number[] = [];
    const vx: Float64Array[] = [];
    const vyList: number[] = [];

    for (let i = 0; i < posIdx.length; i++) {
      const idx = posIdx[i];
      if (i % kFolds === k) {
        vx.push(fx[idx]);
        vyList.push(fy[idx]);
      } else {
        tx.push(fx[idx]);
        tyList.push(fy[idx]);
      }
    }
    for (let i = 0; i < negIdx.length; i++) {
      const idx = negIdx[i];
      if (i % kFolds === k) {
        vx.push(fx[idx]);
        vyList.push(fy[idx]);
      } else {
        tx.push(fx[idx]);
        tyList.push(fy[idx]);
      }
    }
    folds.push({
      tx,
      ty: Uint8Array.from(tyList),
      vx,
      vy: Uint8Array.from(vyList),
    });
  }

  const cvTrials: XGBoostModel['cvTrials'] = [];
  let bestParams = candidates[0];
  let bestCvScore = -1;

  for (let c = 0; c < candidates.length; c++) {
    const params = candidates[c];
    let prSum = 0;
    let f1Sum = 0;

    for (let k = 0; k < kFolds; k++) {
      const fold = folds[k];
      const fit = fitXGBoostCore(
        fold.tx,
        fold.ty,
        null,
        null,
        { ...params, nEstimators: Math.min(18, params.nEstimators) },
        500 + c * 10 + k
      );
      const probs = new Float64Array(fold.vx.length);
      for (let i = 0; i < fold.vx.length; i++) {
        let z = fit.baseScore;
        for (const tree of fit.trees) z += params.learningRate * traverseLeafValue(tree, fold.vx[i]);
        probs[i] = sigmoid(z);
      }
      prSum += computePrAucFast(fold.vy, probs);
      f1Sum += computeF1AtThreshold(fold.vy, probs, 0.5);
    }

    const cvPrAuc = Number((prSum / kFolds).toFixed(4));
    const cvF1 = Number((f1Sum / kFolds).toFixed(4));
    cvTrials.push({
      nEstimators: params.nEstimators,
      maxDepth: params.maxDepth,
      learningRate: params.learningRate,
      subsample: params.subsample,
      colsampleBytree: params.colsampleBytree,
      minChildWeight: params.minChildWeight,
      cvPrAuc,
      cvF1,
    });

    const composite = 0.65 * cvPrAuc + 0.35 * cvF1;
    if (composite > bestCvScore) {
      bestCvScore = composite;
      bestParams = params;
    }
  }

  // Final refit on training set with validation-set early stopping
  const finalFit = fitXGBoostCore(fx, fy, valX, valY, bestParams, 777);

  // Optimize decision threshold strictly on validation set
  const valProbs = new Float64Array(valX.length);
  for (let i = 0; i < valX.length; i++) {
    let z = finalFit.baseScore;
    for (const tree of finalFit.trees) z += bestParams.learningRate * traverseLeafValue(tree, valX[i]);
    valProbs[i] = sigmoid(z);
  }

  let optimalThreshold = 0.5;
  let bestValF1 = -1;
  for (let t = 0.20; t <= 0.85; t += 0.05) {
    const f1 = computeF1AtThreshold(valY, valProbs, t);
    if (f1 > bestValF1) {
      bestValF1 = f1;
      optimalThreshold = Number(t.toFixed(2));
    }
  }

  return {
    modelId: 'tuned_xgboost',
    baseScore: finalFit.baseScore,
    trees: finalFit.trees,
    hyperparams: bestParams,
    cvBestScore: Number(bestCvScore.toFixed(4)),
    cvTrials,
    optimalThreshold,
    featureImportances: finalFit.featureImportances,
    earlyStoppedRound: finalFit.earlyStoppedRound,
    trainingTimeMs: Number((performance.now() - t0).toFixed(2)),
  };
}

export function predictXGBoost(model: XGBoostModel, x: Float64Array[]): Float64Array {
  const out = new Float64Array(x.length);
  const lr = model.hyperparams.learningRate;
  for (let i = 0; i < x.length; i++) {
    let z = model.baseScore;
    const row = x[i];
    for (let t = 0; t < model.trees.length; t++) {
      z += lr * traverseLeafValue(model.trees[t], row);
    }
    out[i] = sigmoid(z);
  }
  return out;
}
