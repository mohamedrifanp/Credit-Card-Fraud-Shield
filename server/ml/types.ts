import { ModelId } from '../config.ts';

export interface ColumnProfile {
  name: string;
  dtype: 'float64' | 'int64';
  missingCount: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  correlationWithClass: number;
}

export interface EdaReport {
  generated: boolean;
  source: 'kaggle_csv' | 'benchmark_ulb' | 'none';
  filename: string;
  shape: { rows: number; columns: number };
  columns: ColumnProfile[];
  missingValuesTotal: number;
  duplicateInvestigation: {
    totalDuplicates: number;
    legitDuplicates: number;
    fraudDuplicates: number;
    duplicateRatePct: number;
    rowsAfterDeduplication: number;
    handlingDecision: string;
  };
  topCorrelations: Array<{ feature: string; correlation: number; direction: 'positive' | 'negative' }>;
  generatedAt: string | null;
}

export interface ImbalanceComparisonEntry {
  strategy: 'original_imbalanced' | 'class_weights' | 'smote';
  strategyLabel: string;
  modelName: string;
  trainRows: number;
  trainFraudCount: number;
  valPrecision: number;
  valRecall: number;
  valF1: number;
  valRocAuc: number;
  valPrAuc: number;
  trainingTimeMs: number;
}

export interface ModelMetadataArtifact {
  bestModelId: ModelId;
  bestModelName: string;
  selectedThreshold: number;
  datasetSource: 'kaggle_csv' | 'benchmark_ulb';
  preprocessingFile: string;
  bestModelFile: string;
  selectionCriterion: string;
  testSetEvaluatedOnce: boolean;
  savedAt: string;
}

export interface DatasetSummary {
  loaded: boolean;
  source: 'kaggle_csv' | 'benchmark_ulb' | 'none';
  filename: string;
  totalRows: number;
  legitimateCount: number;
  fraudCount: number;
  fraudPercentage: number;
  featureCount: number;
  missingValues: number;
  duplicateRows: number;
  amountStats: {
    legitMean: number;
    legitMedian: number;
    legitMax: number;
    fraudMean: number;
    fraudMedian: number;
    fraudMax: number;
  };
  amountHistogram: Array<{
    bucket: string;
    legitimate: number;
    fraudulent: number;
  }>;
  timeTrends: Array<{
    hourBucket: string;
    total: number;
    fraudulent: number;
    fraudRate: number;
    avgAmount: number;
  }>;
  highRiskSamples: Array<{
    rowIndex: number;
    time: number;
    amount: number;
    v14: number;
    v17: number;
    v12: number;
    classLabel: number;
  }>;
  loadedAt: string | null;
}

export interface PreprocessingArtifact {
  featureNames: string[];
  timeMedian: number;
  timeIqr: number;
  amountMedian: number;
  amountIqr: number;
  vFeaturesPreserved: boolean;
  duplicatesRemoved: number;
  trainRows: number;
  valRows: number;
  testRows: number;
  trainFraudCount: number;
  valFraudCount: number;
  testFraudCount: number;
  fittedAt: string;
}

export interface LogisticRegressionModel {
  modelId: 'logistic_regression';
  weights: number[];
  bias: number;
  posWeight: number;
  l2Lambda: number;
  epochs: number;
  trainingTimeMs: number;
}

export interface TreeNode {
  isLeaf: boolean;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  probability?: number;
  leafValue?: number;
  samples?: number;
}

export interface DecisionTreeModel {
  modelId: 'decision_tree';
  root: TreeNode;
  maxDepth: number;
  minSamplesLeaf: number;
  posWeight: number;
  featureImportances: number[];
  trainingTimeMs: number;
}

export interface XGBoostHyperparams {
  nEstimators: number;
  maxDepth: number;
  learningRate: number;
  subsample: number;
  colsampleBytree: number;
  minChildWeight: number;
  regLambda: number;
  scalePosWeight: number;
}

export interface XGBoostModel {
  modelId: 'tuned_xgboost';
  baseScore: number;
  trees: TreeNode[];
  hyperparams: XGBoostHyperparams;
  cvBestScore: number;
  cvTrials: Array<{
    nEstimators: number;
    maxDepth: number;
    learningRate: number;
    subsample: number;
    colsampleBytree: number;
    minChildWeight: number;
    cvPrAuc: number;
    cvF1: number;
  }>;
  optimalThreshold: number;
  featureImportances: number[];
  earlyStoppedRound: number;
  trainingTimeMs: number;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface CurvePoint {
  x: number;
  y: number;
  threshold: number;
}

export interface ThresholdSweepPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  flaggedCount: number;
}

export interface ModelEvaluation {
  modelId: ModelId;
  modelName: string;
  thresholdUsed: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  prAuc: number;
  trainingTimeMs: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: CurvePoint[];
  prCurve: CurvePoint[];
  thresholdSweep: ThresholdSweepPoint[];
  featureImportances: Array<{ feature: string; importance: number }>;
  classificationReport: {
    legitimate: { precision: number; recall: number; f1: number; support: number };
    fraudulent: { precision: number; recall: number; f1: number; support: number };
    macroAvg: { precision: number; recall: number; f1: number; support: number };
  };
}

export interface PCAExperimentResult {
  explainedVarianceRatio: number[];
  cumulativeVariance: number[];
  componentsFor95Pct: number;
  beforePca: {
    precision: number;
    recall: number;
    f1Score: number;
    rocAuc: number;
    prAuc: number;
    dimensions: number;
  };
  afterPca: {
    precision: number;
    recall: number;
    f1Score: number;
    rocAuc: number;
    prAuc: number;
    dimensions: number;
  };
  note: string;
}

export interface DBSCANPoint {
  x: number;
  y: number;
  cluster: number;
  isNoise: boolean;
  actualClass: number;
  amount: number;
}

export interface DBSCANExperimentResult {
  eps: number;
  minSamples: number;
  sampledCount: number;
  clusterCount: number;
  noiseCount: number;
  noisePercentage: number;
  noiseFraudCount: number;
  totalFraudInSample: number;
  noiseFraudRecall: number;
  noiseFraudPrecision: number;
  points: DBSCANPoint[];
  note: string;
}

export interface EvaluationReport {
  trained: boolean;
  datasetSource: 'kaggle_csv' | 'benchmark_ulb' | 'none';
  bestModelId: ModelId | null;
  selectionCriterion: string;
  imbalanceComparison: ImbalanceComparisonEntry[];
  models: Record<ModelId, ModelEvaluation> | null;
  pcaExperiment: PCAExperimentResult | null;
  dbscanExperiment: DBSCANExperimentResult | null;
  cvTrials: XGBoostModel['cvTrials'];
  xgbBestParams: XGBoostHyperparams | null;
  xgbEarlyStoppedRound: number | null;
  trainedAt: string | null;
}

export interface PredictionRecord {
  id: number;
  transaction_reference: string;
  prediction: 'LEGITIMATE' | 'FRAUDULENT';
  fraud_probability: number;
  model_name: string;
  model_id: ModelId;
  risk_level: 'Low' | 'Medium' | 'High' | 'Critical';
  amount: number;
  tx_time: number;
  features_summary: string;
  created_at: string;
}
