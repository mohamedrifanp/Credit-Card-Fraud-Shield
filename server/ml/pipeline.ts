import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import {
  classifyRiskLevel,
  DATASET_PATH,
  DATA_DIR,
  ensureDirectories,
  FEATURE_NAMES,
  MODEL_DISPLAY_NAMES,
  ModelId,
  MODELS_DIR,
  REPORTS_DIR,
} from '../config.ts';
import { getSettings, savePrediction } from '../database.ts';
import { evaluateModelOnTest, selectBestModel, writeModelComparisonCsv } from './evaluate.ts';
import { runDBSCANExperiment, runPCAExperiment } from './experiments.ts';
import {
  predictDecisionTree,
  predictLogisticRegression,
  predictXGBoost,
  trainDecisionTree,
  trainLogisticRegression,
  trainTunedXGBoost,
} from './models.ts';
import {
  fitPreprocessing,
  parseCreditCardCsvFile,
  stratifiedSplit,
  transformDataset,
  transformRow,
  writeBenchmarkUlbCsv,
} from './preprocessing.ts';
import { runImbalanceComparison } from './smote.ts';
import {
  DatasetSummary,
  DecisionTreeModel,
  EdaReport,
  EvaluationReport,
  LogisticRegressionModel,
  ModelMetadataArtifact,
  PredictionRecord,
  PreprocessingArtifact,
  XGBoostModel,
} from './types.ts';

const BENCHMARK_PATH = path.join(DATA_DIR, 'benchmark_ulb.csv');
const EDA_REPORT_PATH = path.join(REPORTS_DIR, 'eda_report.json');
const DATASET_SUMMARY_PATH = path.join(MODELS_DIR, 'dataset_summary.json');
const PREPROCESSING_PATH = path.join(MODELS_DIR, 'preprocessing.json');
const LOGISTIC_PATH = path.join(MODELS_DIR, 'logistic_model.json');
const TREE_PATH = path.join(MODELS_DIR, 'decision_tree.json');
const XGB_PATH = path.join(MODELS_DIR, 'xgboost_model.json');
const BEST_MODEL_PATH = path.join(MODELS_DIR, 'best_model.json');
const MODEL_METADATA_PATH = path.join(MODELS_DIR, 'model_metadata.json');
const EVAL_REPORT_PATH = path.join(MODELS_DIR, 'evaluation_report.json');

interface PipelineState {
  datasetSummary: DatasetSummary;
  edaReport: EdaReport;
  preprocessing: PreprocessingArtifact | null;
  logisticModel: LogisticRegressionModel | null;
  decisionTreeModel: DecisionTreeModel | null;
  xgboostModel: XGBoostModel | null;
  modelMetadata: ModelMetadataArtifact | null;
  evaluationReport: EvaluationReport;
  isTraining: boolean;
}

const EMPTY_SUMMARY: DatasetSummary = {
  loaded: false,
  source: 'none',
  filename: 'creditcard.csv (not found)',
  totalRows: 0,
  legitimateCount: 0,
  fraudCount: 0,
  fraudPercentage: 0,
  featureCount: FEATURE_NAMES.length,
  missingValues: 0,
  duplicateRows: 0,
  amountStats: {
    legitMean: 0,
    legitMedian: 0,
    legitMax: 0,
    fraudMean: 0,
    fraudMedian: 0,
    fraudMax: 0,
  },
  amountHistogram: [],
  timeTrends: [],
  highRiskSamples: [],
  loadedAt: null,
};

const EMPTY_EDA: EdaReport = {
  generated: false,
  source: 'none',
  filename: 'creditcard.csv (not found)',
  shape: { rows: 0, columns: 0 },
  columns: [],
  missingValuesTotal: 0,
  duplicateInvestigation: {
    totalDuplicates: 0,
    legitDuplicates: 0,
    fraudDuplicates: 0,
    duplicateRatePct: 0,
    rowsAfterDeduplication: 0,
    handlingDecision: 'Awaiting dataset inspection.',
  },
  topCorrelations: [],
  generatedAt: null,
};

const EMPTY_REPORT: EvaluationReport = {
  trained: false,
  datasetSource: 'none',
  bestModelId: null,
  selectionCriterion:
    'Composite imbalanced score: 0.45 × PR-AUC + 0.35 × F1-Score + 0.20 × Recall on untouched stratified test split',
  imbalanceComparison: [],
  models: null,
  pcaExperiment: null,
  dbscanExperiment: null,
  cvTrials: [],
  xgbBestParams: null,
  xgbEarlyStoppedRound: null,
  trainedAt: null,
};

const state: PipelineState = {
  datasetSummary: EMPTY_SUMMARY,
  edaReport: EMPTY_EDA,
  preprocessing: null,
  logisticModel: null,
  decisionTreeModel: null,
  xgboostModel: null,
  modelMetadata: null,
  evaluationReport: EMPTY_REPORT,
  isTraining: false,
};

function readJsonIfExists<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
  } catch {
    return null;
  }
}

export function initializePipeline(): void {
  ensureDirectories();
  const cachedSummary = readJsonIfExists<DatasetSummary>(DATASET_SUMMARY_PATH);
  const cachedEda = readJsonIfExists<EdaReport>(EDA_REPORT_PATH);
  const cachedPrep = readJsonIfExists<PreprocessingArtifact>(PREPROCESSING_PATH);
  const cachedLog = readJsonIfExists<LogisticRegressionModel>(LOGISTIC_PATH);
  const cachedTree = readJsonIfExists<DecisionTreeModel>(TREE_PATH);
  const cachedXgb = readJsonIfExists<XGBoostModel>(XGB_PATH);
  const cachedMeta = readJsonIfExists<ModelMetadataArtifact>(MODEL_METADATA_PATH);
  const cachedReport = readJsonIfExists<EvaluationReport>(EVAL_REPORT_PATH);

  if (cachedSummary) state.datasetSummary = cachedSummary;
  if (cachedEda) state.edaReport = cachedEda;
  if (cachedPrep) state.preprocessing = cachedPrep;
  if (cachedLog) state.logisticModel = cachedLog;
  if (cachedTree) state.decisionTreeModel = cachedTree;
  if (cachedXgb) state.xgboostModel = cachedXgb;
  if (cachedMeta) state.modelMetadata = cachedMeta;
  if (cachedReport) state.evaluationReport = cachedReport;
}

export function getPipelineStatus(): {
  datasetSummary: DatasetSummary;
  edaReport: EdaReport;
  evaluationReport: EvaluationReport;
  preprocessing: PreprocessingArtifact | null;
  modelMetadata: ModelMetadataArtifact | null;
  modelsReady: Record<ModelId, boolean>;
  isTraining: boolean;
  kaggleFilePresent: boolean;
} {
  return {
    datasetSummary: state.datasetSummary,
    edaReport: state.edaReport,
    evaluationReport: state.evaluationReport,
    preprocessing: state.preprocessing,
    modelMetadata: state.modelMetadata,
    modelsReady: {
      logistic_regression: state.logisticModel !== null,
      decision_tree: state.decisionTreeModel !== null,
      tuned_xgboost: state.xgboostModel !== null,
    },
    isTraining: state.isTraining,
    kaggleFilePresent: fs.existsSync(DATASET_PATH),
  };
}

/**
 * Phase 1 Standalone Inspection (Data Collection & EDA without training).
 */
export async function inspectDatasetEda(options: {
  useKaggleIfPresent?: boolean;
  generateBenchmark?: boolean;
}): Promise<{ datasetSummary: DatasetSummary; edaReport: EdaReport }> {
  ensureDirectories();
  let targetFile = DATASET_PATH;
  let source: 'kaggle_csv' | 'benchmark_ulb' = 'kaggle_csv';
  let filename = 'creditcard.csv';

  if (options.useKaggleIfPresent && fs.existsSync(DATASET_PATH)) {
    targetFile = DATASET_PATH;
  } else if (options.generateBenchmark) {
    writeBenchmarkUlbCsv(BENCHMARK_PATH, 6000);
    targetFile = BENCHMARK_PATH;
    source = 'benchmark_ulb';
    filename = 'benchmark_ulb.csv (Synthetic ULB-Schema)';
  } else if (fs.existsSync(DATASET_PATH)) {
    targetFile = DATASET_PATH;
  } else {
    throw new Error('Dataset data/creditcard.csv not found.');
  }

  const parsed = await parseCreditCardCsvFile(targetFile, source, filename);
  fs.writeFileSync(DATASET_SUMMARY_PATH, JSON.stringify(parsed.summary), 'utf8');
  fs.writeFileSync(EDA_REPORT_PATH, JSON.stringify(parsed.edaReport, null, 2), 'utf8');

  state.datasetSummary = parsed.summary;
  state.edaReport = parsed.edaReport;
  return { datasetSummary: parsed.summary, edaReport: parsed.edaReport };
}

/**
 * Full End-to-End Phases 1–8 ML Pipeline:
 * Phase 1: EDA & Duplicate Investigation
 * Phase 2: Preprocessing (RobustScaler on Time & Amount only, preserve V1-V28, stratified split)
 * Phase 3: Class Imbalance Comparison (Original vs Class Weights vs SMOTE on train fold -> val eval)
 * Phase 4 & 6: Model Training & XGBoost CV Hyperparameter Tuning + Val Threshold Selection
 * Phase 5: PCA Variance Comparison & DBSCAN Exploratory Anomaly Detection
 * Phase 7: Untouched Held-Out Test Set Evaluation & Best Model Selection
 * Phase 8: Save Best Model, Preprocessing Pipeline, Threshold, Metadata, and Metrics
 */
export async function runFullTrainingPipeline(options: {
  useKaggleIfPresent?: boolean;
  uploadedFilePath?: string;
  uploadedFilename?: string;
  generateBenchmark?: boolean;
}): Promise<{
  datasetSummary: DatasetSummary;
  edaReport: EdaReport;
  evaluationReport: EvaluationReport;
  preprocessing: PreprocessingArtifact;
  modelMetadata: ModelMetadataArtifact;
}> {
  if (state.isTraining) {
    throw new Error('Model training is already in progress. Please wait for completion.');
  }

  state.isTraining = true;
  try {
    ensureDirectories();

    let targetFile = DATASET_PATH;
    let source: 'kaggle_csv' | 'benchmark_ulb' = 'kaggle_csv';
    let filename = 'creditcard.csv';

    if (options.uploadedFilePath && fs.existsSync(options.uploadedFilePath)) {
      targetFile = options.uploadedFilePath;
      source = 'kaggle_csv';
      filename = options.uploadedFilename || 'creditcard.csv';
    } else if (options.useKaggleIfPresent && fs.existsSync(DATASET_PATH)) {
      targetFile = DATASET_PATH;
      source = 'kaggle_csv';
      filename = 'creditcard.csv';
    } else if (options.generateBenchmark) {
      writeBenchmarkUlbCsv(BENCHMARK_PATH, 6000);
      targetFile = BENCHMARK_PATH;
      source = 'benchmark_ulb';
      filename = 'benchmark_ulb.csv (Synthetic ULB-Schema)';
    } else if (fs.existsSync(DATASET_PATH)) {
      targetFile = DATASET_PATH;
      source = 'kaggle_csv';
      filename = 'creditcard.csv';
    } else if (fs.existsSync(BENCHMARK_PATH)) {
      targetFile = BENCHMARK_PATH;
      source = 'benchmark_ulb';
      filename = 'benchmark_ulb.csv (Synthetic ULB-Schema)';
    } else {
      throw new Error(
        'Dataset data/creditcard.csv not found. Upload creditcard.csv from Kaggle or generate the ULB-schema benchmark dataset.'
      );
    }

    // Phase 1: Parse, inspect shape/dtypes, missing values, duplicate investigation, and correlations
    const parsed = await parseCreditCardCsvFile(targetFile, source, filename);

    // Phase 2: Stratified Train (70%) / Validation (15%) / Test (15%) split (zero data leakage)
    const splits = stratifiedSplit(parsed.features, parsed.labels, 0.70, 0.15);

    // Fit RobustScaler ONLY on Time & Amount using training data; preserve V1–V28 PCA features
    const preprocessing = fitPreprocessing(
      splits.trainX,
      splits.trainY,
      splits.valY,
      splits.testY,
      parsed.edaReport.duplicateInvestigation.totalDuplicates
    );
    const trainScaled = transformDataset(splits.trainX, preprocessing);
    const valScaled = transformDataset(splits.valX, preprocessing);
    const testScaled = transformDataset(splits.testX, preprocessing);

    // Phase 3: Class Imbalance Comparison (Original vs Class Weights vs SMOTE on training fold -> Validation eval)
    const imbalanceComparison = runImbalanceComparison(
      trainScaled,
      splits.trainY,
      valScaled,
      splits.valY
    );

    // Phase 4 & 6: Train Logistic Regression, Decision Tree, and Tuned XGBoost (CV + Val Early Stopping + Val Threshold)
    const logisticModel = trainLogisticRegression(trainScaled, splits.trainY, 120, 0.002, true);
    const decisionTreeModel = trainDecisionTree(trainScaled, splits.trainY, 5, 12, true);
    const xgboostModel = trainTunedXGBoost(trainScaled, splits.trainY, valScaled, splits.valY);

    // Phase 7: Final evaluation ONCE on the untouched held-out test set
    const logProbs = predictLogisticRegression(logisticModel, testScaled);
    const treeProbs = predictDecisionTree(decisionTreeModel, testScaled);
    const xgbProbs = predictXGBoost(xgboostModel, testScaled);

    const logEval = evaluateModelOnTest(
      'logistic_regression',
      splits.testY,
      logProbs,
      0.5,
      logisticModel.trainingTimeMs,
      logisticModel.weights
    );
    const treeEval = evaluateModelOnTest(
      'decision_tree',
      splits.testY,
      treeProbs,
      0.5,
      decisionTreeModel.trainingTimeMs,
      decisionTreeModel.featureImportances
    );
    const xgbEval = evaluateModelOnTest(
      'tuned_xgboost',
      splits.testY,
      xgbProbs,
      xgboostModel.optimalThreshold,
      xgboostModel.trainingTimeMs,
      xgboostModel.featureImportances
    );

    const modelsEval = {
      logistic_regression: logEval,
      decision_tree: treeEval,
      tuned_xgboost: xgbEval,
    };

    const bestModelId = selectBestModel(modelsEval);

    // Phase 5: PCA Dimensionality Reduction Experiment & DBSCAN Exploratory Anomaly Detection
    const { pcaResult, eigenvectors } = runPCAExperiment(
      trainScaled,
      splits.trainY,
      testScaled,
      splits.testY,
      logEval
    );

    const dbscanExperiment = runDBSCANExperiment(
      testScaled,
      splits.testX,
      splits.testY,
      eigenvectors
    );

    const nowIso = new Date().toISOString();
    const evaluationReport: EvaluationReport = {
      trained: true,
      datasetSource: source,
      bestModelId,
      selectionCriterion:
        'Composite imbalanced score: 0.45 × PR-AUC + 0.35 × F1-Score + 0.20 × Recall on untouched stratified test split',
      imbalanceComparison,
      models: modelsEval,
      pcaExperiment: pcaResult,
      dbscanExperiment,
      cvTrials: xgboostModel.cvTrials,
      xgbBestParams: xgboostModel.hyperparams,
      xgbEarlyStoppedRound: xgboostModel.earlyStoppedRound,
      trainedAt: nowIso,
    };

    const bestArtifact =
      bestModelId === 'logistic_regression'
        ? logisticModel
        : bestModelId === 'decision_tree'
        ? decisionTreeModel
        : xgboostModel;

    const modelMetadata: ModelMetadataArtifact = {
      bestModelId,
      bestModelName: MODEL_DISPLAY_NAMES[bestModelId],
      selectedThreshold: modelsEval[bestModelId].thresholdUsed,
      datasetSource: source,
      preprocessingFile: 'models/preprocessing.json',
      bestModelFile: 'models/best_model.json',
      selectionCriterion: evaluationReport.selectionCriterion,
      testSetEvaluatedOnce: true,
      savedAt: nowIso,
    };

    // Phase 8: Persist best model, preprocessing pipeline, threshold, metadata, EDA & evaluation reports
    fs.writeFileSync(DATASET_SUMMARY_PATH, JSON.stringify(parsed.summary), 'utf8');
    fs.writeFileSync(EDA_REPORT_PATH, JSON.stringify(parsed.edaReport, null, 2), 'utf8');
    fs.writeFileSync(PREPROCESSING_PATH, JSON.stringify(preprocessing, null, 2), 'utf8');
    fs.writeFileSync(LOGISTIC_PATH, JSON.stringify(logisticModel), 'utf8');
    fs.writeFileSync(TREE_PATH, JSON.stringify(decisionTreeModel), 'utf8');
    fs.writeFileSync(XGB_PATH, JSON.stringify(xgboostModel), 'utf8');
    fs.writeFileSync(BEST_MODEL_PATH, JSON.stringify(bestArtifact), 'utf8');
    fs.writeFileSync(MODEL_METADATA_PATH, JSON.stringify(modelMetadata, null, 2), 'utf8');
    fs.writeFileSync(EVAL_REPORT_PATH, JSON.stringify(evaluationReport), 'utf8');
    writeModelComparisonCsv(modelsEval);

    state.datasetSummary = parsed.summary;
    state.edaReport = parsed.edaReport;
    state.preprocessing = preprocessing;
    state.logisticModel = logisticModel;
    state.decisionTreeModel = decisionTreeModel;
    state.xgboostModel = xgboostModel;
    state.modelMetadata = modelMetadata;
    state.evaluationReport = evaluationReport;

    return {
      datasetSummary: parsed.summary,
      edaReport: parsed.edaReport,
      evaluationReport,
      preprocessing,
      modelMetadata,
    };
  } finally {
    state.isTraining = false;
  }
}

function scoreSingleVector(scaled: Float64Array, modelId: ModelId): number {
  if (modelId === 'logistic_regression') {
    if (!state.logisticModel) throw new Error('Logistic Regression model is not trained yet.');
    return predictLogisticRegression(state.logisticModel, [scaled])[0];
  }
  if (modelId === 'decision_tree') {
    if (!state.decisionTreeModel) throw new Error('Decision Tree model is not trained yet.');
    return predictDecisionTree(state.decisionTreeModel, [scaled])[0];
  }
  if (!state.xgboostModel) throw new Error('Tuned XGBoost model is not trained yet.');
  return predictXGBoost(state.xgboostModel, [scaled])[0];
}

function buildFeatureSummary(raw: Float64Array, scaled: Float64Array, modelId: ModelId): string {
  const evalImportances = state.evaluationReport.models?.[modelId]?.featureImportances;
  const impMap = new Map<string, number>();
  if (evalImportances) {
    for (const item of evalImportances) impMap.set(item.feature, item.importance);
  }

  const drivers = FEATURE_NAMES.map((name, idx) => ({
    feature: name,
    raw: Number(raw[idx].toFixed(4)),
    zScore: Number(scaled[idx].toFixed(3)),
    score: Math.abs(scaled[idx]) * (0.2 + (impMap.get(name) ?? 0.03)),
  }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

  return JSON.stringify({ topDrivers: drivers });
}

export interface SinglePredictionOutput {
  record: PredictionRecord;
  deduplicated: boolean;
  confidenceLabel: string;
  thresholdUsed: number;
  topDrivers: Array<{ feature: string; raw: number; zScore: number }>;
}

export function predictSingleTransaction(
  rawFeatures: Float64Array,
  options?: {
    modelId?: ModelId;
    threshold?: number;
    transactionReference?: string;
  }
): SinglePredictionOutput {
  if (!state.preprocessing) {
    throw new Error('Models are not trained yet. Load a dataset and run training first.');
  }

  const settings = getSettings();
  const modelId = options?.modelId || settings.activeModel;
  const threshold = options?.threshold ?? settings.threshold;

  const scaled = transformRow(rawFeatures, state.preprocessing);
  const prob = scoreSingleVector(scaled, modelId);
  const clampedProb = Number(Math.min(0.9999, Math.max(0.0001, prob)).toFixed(4));
  const prediction: 'LEGITIMATE' | 'FRAUDULENT' =
    clampedProb >= threshold ? 'FRAUDULENT' : 'LEGITIMATE';
  const riskLevel = classifyRiskLevel(clampedProb, threshold);

  const ref =
    options?.transactionReference && options.transactionReference.trim().length > 0
      ? options.transactionReference.trim()
      : `TX-${Date.now().toString(36).toUpperCase()}-${crypto.randomBytes(2).toString('hex').toUpperCase()}`;

  const summaryJson = buildFeatureSummary(rawFeatures, scaled, modelId);
  const { record, deduplicated } = savePrediction({
    transaction_reference: ref,
    prediction,
    fraud_probability: clampedProb,
    model_name: MODEL_DISPLAY_NAMES[modelId],
    model_id: modelId,
    risk_level: riskLevel,
    amount: Number(rawFeatures[29].toFixed(2)),
    tx_time: Number(rawFeatures[0].toFixed(2)),
    features_summary: summaryJson,
  });

  const margin = Math.abs(clampedProb - threshold);
  const confidenceLabel =
    margin >= 0.30
      ? 'High Separation from Threshold'
      : margin >= 0.12
      ? 'Moderate Separation'
      : 'Near Decision Boundary — Manual Review Recommended';

  const parsedSummary = JSON.parse(summaryJson) as {
    topDrivers: Array<{ feature: string; raw: number; zScore: number }>;
  };

  return {
    record,
    deduplicated,
    confidenceLabel,
    thresholdUsed: threshold,
    topDrivers: parsedSummary.topDrivers,
  };
}

export function predictBatchTransactions(
  rows: Array<{ reference?: string; features: Float64Array }>,
  options?: { modelId?: ModelId; threshold?: number }
): {
  results: SinglePredictionOutput[];
  summary: {
    total: number;
    fraudulent: number;
    legitimate: number;
    deduplicated: number;
  };
} {
  if (!state.preprocessing) {
    throw new Error('Models are not trained yet. Load a dataset and run training first.');
  }
  const settings = getSettings();
  const modelId = options?.modelId || settings.activeModel;
  const threshold = options?.threshold ?? settings.threshold;

  const results: SinglePredictionOutput[] = [];
  let fraudulent = 0;
  let legitimate = 0;
  let deduplicated = 0;

  for (let i = 0; i < rows.length; i++) {
    const item = rows[i];
    const deterministicHash = crypto
      .createHash('sha256')
      .update(
        Array.from(item.features)
          .map((v) => v.toFixed(4))
          .join(',') + `|${modelId}|${threshold.toFixed(2)}`
      )
      .digest('hex')
      .slice(0, 10)
      .toUpperCase();

    const ref = item.reference || `BATCH-${deterministicHash}`;
    const out = predictSingleTransaction(item.features, {
      modelId,
      threshold,
      transactionReference: ref,
    });

    if (out.record.prediction === 'FRAUDULENT') fraudulent++;
    else legitimate++;
    if (out.deduplicated) deduplicated++;
    results.push(out);
  }

  return {
    results,
    summary: {
      total: results.length,
      fraudulent,
      legitimate,
      deduplicated,
    },
  };
}
