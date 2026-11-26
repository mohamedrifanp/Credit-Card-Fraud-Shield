import { FEATURE_NAMES } from '../config.ts';
import { evaluateModelOnTest } from './evaluate.ts';
import {
  predictDecisionTree,
  predictLogisticRegression,
  trainDecisionTree,
  trainLogisticRegression,
} from './models.ts';
import { mulberry32 } from './preprocessing.ts';
import { ImbalanceComparisonEntry } from './types.ts';

/**
 * Phase 3: Synthetic Minority Over-sampling Technique (SMOTE)
 * Applied STRICTLY to training data only — never to validation or test splits.
 * Finds k nearest neighbors among minority (Class = 1) samples in Euclidean space
 * and interpolates synthetic samples until minority prevalence reaches targetRatio.
 */
export function applySmoteToTraining(
  trainX: Float64Array[],
  trainY: Uint8Array,
  targetRatio = 0.20,
  kNeighbors = 5,
  seed = 42
): { x: Float64Array[]; y: Uint8Array; syntheticAdded: number } {
  const rand = mulberry32(seed);
  const d = FEATURE_NAMES.length;

  const minorityRows: Float64Array[] = [];
  let majorityCount = 0;
  for (let i = 0; i < trainY.length; i++) {
    if (trainY[i] === 1) minorityRows.push(trainX[i]);
    else majorityCount++;
  }

  const m = minorityRows.length;
  if (m < 2) {
    return { x: trainX, y: trainY, syntheticAdded: 0 };
  }

  // Cap majority reference count for controlled synthetic generation (8 GB RAM friendly)
  const effectiveMajority = Math.min(majorityCount, 12000);
  const targetMinority = Math.max(
    m,
    Math.min(4500, Math.round((targetRatio / (1 - targetRatio)) * effectiveMajority))
  );
  const needed = Math.max(0, targetMinority - m);
  if (needed === 0) {
    return { x: trainX, y: trainY, syntheticAdded: 0 };
  }

  const k = Math.min(kNeighbors, m - 1);
  // Precompute k nearest minority neighbors for each minority point
  const neighbors: number[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    const xi = minorityRows[i];
    const dists: Array<{ idx: number; distSq: number }> = [];
    for (let j = 0; j < m; j++) {
      if (i === j) continue;
      const xj = minorityRows[j];
      let sq = 0;
      for (let f = 0; f < d; f++) {
        const diff = xi[f] - xj[f];
        sq += diff * diff;
      }
      dists.push({ idx: j, distSq: sq });
    }
    dists.sort((a, b) => a.distSq - b.distSq);
    neighbors[i] = dists.slice(0, k).map((item) => item.idx);
  }

  const outX: Float64Array[] = trainX.slice();
  const outYList: number[] = Array.from(trainY);

  for (let s = 0; s < needed; s++) {
    const baseIdx = s % m;
    const nnList = neighbors[baseIdx];
    const nnIdx = nnList[Math.floor(rand() * nnList.length)];
    const alpha = rand();

    const baseVec = minorityRows[baseIdx];
    const nnVec = minorityRows[nnIdx];
    const synth = new Float64Array(d);
    for (let f = 0; f < d; f++) {
      synth[f] = baseVec[f] + alpha * (nnVec[f] - baseVec[f]);
    }
    outX.push(synth);
    outYList.push(1);
  }

  return {
    x: outX,
    y: Uint8Array.from(outYList),
    syntheticAdded: needed,
  };
}

/**
 * Evaluates Original Imbalanced vs Class Weights vs SMOTE strictly on the Validation Split,
 * keeping the held-out Test Set untouched until Phase 7 final evaluation.
 */
export function runImbalanceComparison(
  trainX: Float64Array[],
  trainY: Uint8Array,
  valX: Float64Array[],
  valY: Uint8Array
): ImbalanceComparisonEntry[] {
  const entries: ImbalanceComparisonEntry[] = [];

  let origFraud = 0;
  for (let i = 0; i < trainY.length; i++) if (trainY[i] === 1) origFraud++;

  // Strategy 1: Original Imbalanced (posWeight = 1.0)
  const origLog = trainLogisticRegression(trainX, trainY, 100, 0.002, false);
  const origLogEval = evaluateModelOnTest(
    'logistic_regression',
    valY,
    predictLogisticRegression(origLog, valX),
    0.5,
    origLog.trainingTimeMs,
    origLog.weights
  );
  entries.push({
    strategy: 'original_imbalanced',
    strategyLabel: 'Original Imbalanced',
    modelName: 'Logistic Regression',
    trainRows: trainX.length,
    trainFraudCount: origFraud,
    valPrecision: origLogEval.precision,
    valRecall: origLogEval.recall,
    valF1: origLogEval.f1Score,
    valRocAuc: origLogEval.rocAuc,
    valPrAuc: origLogEval.prAuc,
    trainingTimeMs: origLog.trainingTimeMs,
  });

  const origTree = trainDecisionTree(trainX, trainY, 5, 12, false);
  const origTreeEval = evaluateModelOnTest(
    'decision_tree',
    valY,
    predictDecisionTree(origTree, valX),
    0.5,
    origTree.trainingTimeMs,
    origTree.featureImportances
  );
  entries.push({
    strategy: 'original_imbalanced',
    strategyLabel: 'Original Imbalanced',
    modelName: 'Decision Tree',
    trainRows: trainX.length,
    trainFraudCount: origFraud,
    valPrecision: origTreeEval.precision,
    valRecall: origTreeEval.recall,
    valF1: origTreeEval.f1Score,
    valRocAuc: origTreeEval.rocAuc,
    valPrAuc: origTreeEval.prAuc,
    trainingTimeMs: origTree.trainingTimeMs,
  });

  // Strategy 2: Class Weights
  const cwLog = trainLogisticRegression(trainX, trainY, 100, 0.002, true);
  const cwLogEval = evaluateModelOnTest(
    'logistic_regression',
    valY,
    predictLogisticRegression(cwLog, valX),
    0.5,
    cwLog.trainingTimeMs,
    cwLog.weights
  );
  entries.push({
    strategy: 'class_weights',
    strategyLabel: 'Class Weights',
    modelName: 'Logistic Regression',
    trainRows: trainX.length,
    trainFraudCount: origFraud,
    valPrecision: cwLogEval.precision,
    valRecall: cwLogEval.recall,
    valF1: cwLogEval.f1Score,
    valRocAuc: cwLogEval.rocAuc,
    valPrAuc: cwLogEval.prAuc,
    trainingTimeMs: cwLog.trainingTimeMs,
  });

  const cwTree = trainDecisionTree(trainX, trainY, 5, 12, true);
  const cwTreeEval = evaluateModelOnTest(
    'decision_tree',
    valY,
    predictDecisionTree(cwTree, valX),
    0.5,
    cwTree.trainingTimeMs,
    cwTree.featureImportances
  );
  entries.push({
    strategy: 'class_weights',
    strategyLabel: 'Class Weights',
    modelName: 'Decision Tree',
    trainRows: trainX.length,
    trainFraudCount: origFraud,
    valPrecision: cwTreeEval.precision,
    valRecall: cwTreeEval.recall,
    valF1: cwTreeEval.f1Score,
    valRocAuc: cwTreeEval.rocAuc,
    valPrAuc: cwTreeEval.prAuc,
    trainingTimeMs: cwTree.trainingTimeMs,
  });

  // Strategy 3: SMOTE (applied ONLY to training fold)
  const smoteData = applySmoteToTraining(trainX, trainY, 0.20, 5, 42);
  let smoteFraud = 0;
  for (let i = 0; i < smoteData.y.length; i++) if (smoteData.y[i] === 1) smoteFraud++;

  const smLog = trainLogisticRegression(smoteData.x, smoteData.y, 100, 0.002, false);
  const smLogEval = evaluateModelOnTest(
    'logistic_regression',
    valY,
    predictLogisticRegression(smLog, valX),
    0.5,
    smLog.trainingTimeMs,
    smLog.weights
  );
  entries.push({
    strategy: 'smote',
    strategyLabel: 'SMOTE (Train Fold Only)',
    modelName: 'Logistic Regression',
    trainRows: smoteData.x.length,
    trainFraudCount: smoteFraud,
    valPrecision: smLogEval.precision,
    valRecall: smLogEval.recall,
    valF1: smLogEval.f1Score,
    valRocAuc: smLogEval.rocAuc,
    valPrAuc: smLogEval.prAuc,
    trainingTimeMs: smLog.trainingTimeMs,
  });

  const smTree = trainDecisionTree(smoteData.x, smoteData.y, 5, 12, false);
  const smTreeEval = evaluateModelOnTest(
    'decision_tree',
    valY,
    predictDecisionTree(smTree, valX),
    0.5,
    smTree.trainingTimeMs,
    smTree.featureImportances
  );
  entries.push({
    strategy: 'smote',
    strategyLabel: 'SMOTE (Train Fold Only)',
    modelName: 'Decision Tree',
    trainRows: smoteData.x.length,
    trainFraudCount: smoteFraud,
    valPrecision: smTreeEval.precision,
    valRecall: smTreeEval.recall,
    valF1: smTreeEval.f1Score,
    valRocAuc: smTreeEval.rocAuc,
    valPrAuc: smTreeEval.prAuc,
    trainingTimeMs: smTree.trainingTimeMs,
  });

  return entries;
}
