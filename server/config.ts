import fs from 'fs';
import path from 'path';

export const ROOT_DIR = process.cwd();
export const DATA_DIR = path.join(ROOT_DIR, 'data');
export const MODELS_DIR = path.join(ROOT_DIR, 'models');
export const DATABASE_DIR = path.join(ROOT_DIR, 'database');
export const REPORTS_DIR = path.join(ROOT_DIR, 'reports');

export const DATASET_PATH = path.join(DATA_DIR, 'creditcard.csv');
export const DB_PATH = path.join(DATABASE_DIR, 'transactions.db');
export const COMPARISON_CSV_PATH = path.join(REPORTS_DIR, 'model_comparison.csv');

export const FEATURE_NAMES = [
  'Time',
  'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
  'Amount',
] as const;

export const MODEL_IDS = ['logistic_regression', 'decision_tree', 'tuned_xgboost'] as const;
export type ModelId = (typeof MODEL_IDS)[number];

export const MODEL_DISPLAY_NAMES: Record<ModelId, string> = {
  logistic_regression: 'Logistic Regression',
  decision_tree: 'Decision Tree',
  tuned_xgboost: 'Tuned XGBoost',
};

export function ensureDirectories(): void {
  for (const dir of [DATA_DIR, MODELS_DIR, DATABASE_DIR, REPORTS_DIR]) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }
}

export function classifyRiskLevel(probability: number, threshold: number): 'Low' | 'Medium' | 'High' | 'Critical' {
  if (probability >= Math.max(0.85, threshold + 0.25)) return 'Critical';
  if (probability >= threshold) return 'High';
  if (probability >= threshold * 0.55) return 'Medium';
  return 'Low';
}
