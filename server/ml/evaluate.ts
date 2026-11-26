import fs from 'fs';
import { COMPARISON_CSV_PATH, FEATURE_NAMES, MODEL_DISPLAY_NAMES, ModelId } from '../config.ts';
import { computePrAucFast } from './models.ts';
import {
  ConfusionMatrix,
  CurvePoint,
  ModelEvaluation,
  ThresholdSweepPoint,
} from './types.ts';

function confusionAtThreshold(
  yTrue: Uint8Array,
  probs: Float64Array,
  threshold: number
): ConfusionMatrix {
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const pred = probs[i] >= threshold ? 1 : 0;
    const actual = yTrue[i];
    if (pred === 1 && actual === 1) tp++;
    else if (pred === 1 && actual === 0) fp++;
    else if (pred === 0 && actual === 1) fn++;
    else tn++;
  }
  return { tn, fp, fn, tp };
}

function computeRocCurveAndAuc(
  yTrue: Uint8Array,
  probs: Float64Array
): { rocAuc: number; rocCurve: CurvePoint[] } {
  const n = yTrue.length;
  let totalPos = 0;
  for (let i = 0; i < n; i++) if (yTrue[i] === 1) totalPos++;
  const totalNeg = n - totalPos;
  if (totalPos === 0 || totalNeg === 0) {
    return { rocAuc: 0.5, rocCurve: [{ x: 0, y: 0, threshold: 1 }, { x: 1, y: 1, threshold: 0 }] };
  }

  const order = new Array<number>(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => probs[b] - probs[a]);

  let tp = 0;
  let fp = 0;
  let prevFpr = 0;
  let prevTpr = 0;
  let auc = 0;

  const fullPoints: CurvePoint[] = [{ x: 0, y: 0, threshold: 1 }];
  for (let i = 0; i < n; i++) {
    const idx = order[i];
    if (yTrue[idx] === 1) tp++;
    else fp++;
    const fpr = fp / totalNeg;
    const tpr = tp / totalPos;
    auc += (fpr - prevFpr) * ((tpr + prevTpr) / 2);
    prevFpr = fpr;
    prevTpr = tpr;
    fullPoints.push({
      x: Number(fpr.toFixed(4)),
      y: Number(tpr.toFixed(4)),
      threshold: Number(probs[idx].toFixed(4)),
    });
  }

  // Downsample curve cleanly to <= 80 points while preserving endpoints
  const maxPts = 80;
  if (fullPoints.length <= maxPts) {
    return { rocAuc: Number(auc.toFixed(4)), rocCurve: fullPoints };
  }
  const sampled: CurvePoint[] = [];
  for (let k = 0; k < maxPts; k++) {
    const idx = Math.min(fullPoints.length - 1, Math.round((k / (maxPts - 1)) * (fullPoints.length - 1)));
    sampled.push(fullPoints[idx]);
  }
  return { rocAuc: Number(auc.toFixed(4)), rocCurve: sampled };
}

function computePrCurve(yTrue: Uint8Array, probs: Float64Array): CurvePoint[] {
  const steps = 60;
  const points: CurvePoint[] = [];
  for (let s = steps; s >= 0; s--) {
    const thresh = s / steps;
    const cm = confusionAtThreshold(yTrue, probs, thresh);
    const recall = cm.tp + cm.fn > 0 ? cm.tp / (cm.tp + cm.fn) : 0;
    const precision = cm.tp + cm.fp > 0 ? cm.tp / (cm.tp + cm.fp) : 1;
    points.push({
      x: Number(recall.toFixed(4)),
      y: Number(precision.toFixed(4)),
      threshold: Number(thresh.toFixed(3)),
    });
  }
  points.sort((a, b) => a.x - b.x);
  return points;
}

export function evaluateModelOnTest(
  modelId: ModelId,
  yTrue: Uint8Array,
  probs: Float64Array,
  threshold: number,
  trainingTimeMs: number,
  rawImportances: number[]
): ModelEvaluation {
  const cm = confusionAtThreshold(yTrue, probs, threshold);
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const accuracy = total > 0 ? (cm.tp + cm.tn) / total : 0;
  const precision = cm.tp + cm.fp > 0 ? cm.tp / (cm.tp + cm.fp) : 0;
  const recall = cm.tp + cm.fn > 0 ? cm.tp / (cm.tp + cm.fn) : 0;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // Legitimate class (Class 0) metrics for classification report
  const legitPrecision = cm.tn + cm.fn > 0 ? cm.tn / (cm.tn + cm.fn) : 0;
  const legitRecall = cm.tn + cm.fp > 0 ? cm.tn / (cm.tn + cm.fp) : 0;
  const legitF1 =
    legitPrecision + legitRecall > 0
      ? (2 * legitPrecision * legitRecall) / (legitPrecision + legitRecall)
      : 0;

  const { rocAuc, rocCurve } = computeRocCurveAndAuc(yTrue, probs);
  const prAuc = Number(computePrAucFast(yTrue, probs).toFixed(4));
  const prCurve = computePrCurve(yTrue, probs);

  const thresholdSweep: ThresholdSweepPoint[] = [];
  for (let t = 0.1; t <= 0.951; t += 0.05) {
    const th = Number(t.toFixed(2));
    const c = confusionAtThreshold(yTrue, probs, th);
    const p = c.tp + c.fp > 0 ? c.tp / (c.tp + c.fp) : 0;
    const r = c.tp + c.fn > 0 ? c.tp / (c.tp + c.fn) : 0;
    const f1 = p + r > 0 ? (2 * p * r) / (p + r) : 0;
    thresholdSweep.push({
      threshold: th,
      precision: Number(p.toFixed(4)),
      recall: Number(r.toFixed(4)),
      f1: Number(f1.toFixed(4)),
      flaggedCount: c.tp + c.fp,
    });
  }

  // Normalize feature importances
  const absImp = rawImportances.map((v) => Math.abs(v));
  const sumImp = absImp.reduce((a, b) => a + b, 0) || 1;
  const featureImportances = FEATURE_NAMES.map((feat, idx) => ({
    feature: feat,
    importance: Number(((absImp[idx] || 0) / sumImp).toFixed(4)),
  })).sort((a, b) => b.importance - a.importance);

  const legitSupport = cm.tn + cm.fp;
  const fraudSupport = cm.tp + cm.fn;

  return {
    modelId,
    modelName: MODEL_DISPLAY_NAMES[modelId],
    thresholdUsed: Number(threshold.toFixed(2)),
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1Score: Number(f1Score.toFixed(4)),
    rocAuc,
    prAuc,
    trainingTimeMs,
    confusionMatrix: cm,
    rocCurve,
    prCurve,
    thresholdSweep,
    featureImportances,
    classificationReport: {
      legitimate: {
        precision: Number(legitPrecision.toFixed(4)),
        recall: Number(legitRecall.toFixed(4)),
        f1: Number(legitF1.toFixed(4)),
        support: legitSupport,
      },
      fraudulent: {
        precision: Number(precision.toFixed(4)),
        recall: Number(recall.toFixed(4)),
        f1: Number(f1Score.toFixed(4)),
        support: fraudSupport,
      },
      macroAvg: {
        precision: Number(((legitPrecision + precision) / 2).toFixed(4)),
        recall: Number(((legitRecall + recall) / 2).toFixed(4)),
        f1: Number(((legitF1 + f1Score) / 2).toFixed(4)),
        support: legitSupport + fraudSupport,
      },
    },
  };
}

export function selectBestModel(models: Record<ModelId, ModelEvaluation>): ModelId {
  let bestId: ModelId = 'tuned_xgboost';
  let bestComposite = -1;
  for (const id of Object.keys(models) as ModelId[]) {
    const m = models[id];
    // Imbalanced classification priority: PR-AUC (0.45) + F1 (0.35) + Recall (0.20)
    const score = 0.45 * m.prAuc + 0.35 * m.f1Score + 0.20 * m.recall;
    if (score > bestComposite) {
      bestComposite = score;
      bestId = id;
    }
  }
  return bestId;
}

export function writeModelComparisonCsv(models: Record<ModelId, ModelEvaluation>): void {
  const header = [
    'model_id',
    'model_name',
    'threshold',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'pr_auc',
    'training_time_ms',
    'tp',
    'fp',
    'fn',
    'tn',
  ].join(',');

  const rows = (Object.values(models) as ModelEvaluation[]).map((m) =>
    [
      m.modelId,
      `"${m.modelName}"`,
      m.thresholdUsed,
      m.accuracy,
      m.precision,
      m.recall,
      m.f1Score,
      m.rocAuc,
      m.prAuc,
      m.trainingTimeMs,
      m.confusionMatrix.tp,
      m.confusionMatrix.fp,
      m.confusionMatrix.fn,
      m.confusionMatrix.tn,
    ].join(',')
  );

  fs.writeFileSync(COMPARISON_CSV_PATH, [header, ...rows].join('\n'), 'utf8');
}
