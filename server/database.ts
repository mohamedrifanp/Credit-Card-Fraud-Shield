import { DatabaseSync } from 'node:sqlite';
import { DB_PATH, ensureDirectories, ModelId, MODEL_IDS } from './config.ts';
import { PredictionRecord } from './ml/types.ts';

let db: DatabaseSync | null = null;

export function getDb(): DatabaseSync {
  if (db) return db;
  ensureDirectories();
  db = new DatabaseSync(DB_PATH);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      transaction_reference TEXT NOT NULL UNIQUE,
      prediction TEXT NOT NULL CHECK(prediction IN ('LEGITIMATE', 'FRAUDULENT')),
      fraud_probability REAL NOT NULL,
      model_name TEXT NOT NULL,
      model_id TEXT NOT NULL,
      risk_level TEXT NOT NULL CHECK(risk_level IN ('Low', 'Medium', 'High', 'Critical')),
      amount REAL NOT NULL DEFAULT 0,
      tx_time REAL NOT NULL DEFAULT 0,
      features_summary TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(prediction);
    CREATE INDEX IF NOT EXISTS idx_predictions_amount ON predictions(amount);

    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
  `);

  const now = new Date().toISOString();
  const insertSetting = db.prepare(
    'INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)'
  );
  insertSetting.run('active_model', 'tuned_xgboost', now);
  insertSetting.run('threshold', '0.50', now);

  return db;
}

export interface AppSettings {
  activeModel: ModelId;
  threshold: number;
  updatedAt: string;
}

export function getSettings(): AppSettings {
  const database = getDb();
  const rows = database.prepare('SELECT key, value, updated_at FROM settings').all() as Array<{
    key: string;
    value: string;
    updated_at: string;
  }>;

  let activeModel: ModelId = 'tuned_xgboost';
  let threshold = 0.5;
  let updatedAt = new Date().toISOString();

  for (const row of rows) {
    if (row.key === 'active_model' && (MODEL_IDS as readonly string[]).includes(row.value)) {
      activeModel = row.value as ModelId;
      updatedAt = row.updated_at;
    }
    if (row.key === 'threshold') {
      const parsed = Number(row.value);
      if (Number.isFinite(parsed) && parsed >= 0.05 && parsed <= 0.95) {
        threshold = parsed;
      }
    }
  }

  return { activeModel, threshold, updatedAt };
}

export function updateSettings(activeModel: ModelId, threshold: number): AppSettings {
  if (!(MODEL_IDS as readonly string[]).includes(activeModel)) {
    throw new Error('Invalid model selection.');
  }
  if (!Number.isFinite(threshold) || threshold < 0.05 || threshold > 0.95) {
    throw new Error('Classification threshold must be between 0.05 and 0.95.');
  }

  const database = getDb();
  const now = new Date().toISOString();
  const stmt = database.prepare(
    `INSERT INTO settings (key, value, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`
  );
  stmt.run('active_model', activeModel, now);
  stmt.run('threshold', threshold.toFixed(4), now);

  return { activeModel, threshold: Number(threshold.toFixed(4)), updatedAt: now };
}

export interface InsertPredictionInput {
  transaction_reference: string;
  prediction: 'LEGITIMATE' | 'FRAUDULENT';
  fraud_probability: number;
  model_name: string;
  model_id: ModelId;
  risk_level: 'Low' | 'Medium' | 'High' | 'Critical';
  amount: number;
  tx_time: number;
  features_summary: string;
}

export function savePrediction(input: InsertPredictionInput): { record: PredictionRecord; deduplicated: boolean } {
  const database = getDb();
  const now = new Date().toISOString();

  const existing = database
    .prepare('SELECT * FROM predictions WHERE transaction_reference = ?')
    .get(input.transaction_reference) as unknown as PredictionRecord | undefined;

  if (existing) {
    return { record: existing, deduplicated: true };
  }

  const stmt = database.prepare(`
    INSERT INTO predictions (
      transaction_reference, prediction, fraud_probability,
      model_name, model_id, risk_level, amount, tx_time, features_summary, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const result = stmt.run(
    input.transaction_reference,
    input.prediction,
    input.fraud_probability,
    input.model_name,
    input.model_id,
    input.risk_level,
    input.amount,
    input.tx_time,
    input.features_summary,
    now
  );

  const created = database
    .prepare('SELECT * FROM predictions WHERE id = ?')
    .get(Number(result.lastInsertRowid)) as unknown as PredictionRecord;

  return { record: created, deduplicated: false };
}

export interface QueryHistoryParams {
  search?: string;
  status?: 'ALL' | 'LEGITIMATE' | 'FRAUDULENT';
  dateFrom?: string;
  dateTo?: string;
  sortBy?: 'created_at' | 'amount' | 'fraud_probability';
  sortDir?: 'asc' | 'desc';
  limit?: number;
}

export function queryPredictions(params: QueryHistoryParams = {}): PredictionRecord[] {
  const database = getDb();
  const conditions: string[] = [];
  const values: Array<string | number> = [];

  if (params.search && params.search.trim().length > 0) {
    const pattern = `%${params.search.trim()}%`;
    conditions.push('(transaction_reference LIKE ? OR model_name LIKE ? OR risk_level LIKE ?)');
    values.push(pattern, pattern, pattern);
  }

  if (params.status && (params.status === 'LEGITIMATE' || params.status === 'FRAUDULENT')) {
    conditions.push('prediction = ?');
    values.push(params.status);
  }

  if (params.dateFrom && params.dateFrom.trim().length > 0) {
    conditions.push('substr(created_at, 1, 10) >= ?');
    values.push(params.dateFrom.trim());
  }

  if (params.dateTo && params.dateTo.trim().length > 0) {
    conditions.push('substr(created_at, 1, 10) <= ?');
    values.push(params.dateTo.trim());
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
  const allowedSortCols: Record<string, string> = {
    created_at: 'created_at',
    amount: 'amount',
    fraud_probability: 'fraud_probability',
  };
  const sortCol = allowedSortCols[params.sortBy || 'created_at'] || 'created_at';
  const sortDir = params.sortDir === 'asc' ? 'ASC' : 'DESC';
  const limit = Math.min(Math.max(1, Number(params.limit || 200)), 1000);

  const sql = `SELECT * FROM predictions ${whereClause} ORDER BY ${sortCol} ${sortDir}, id DESC LIMIT ?`;
  values.push(limit);

  return database.prepare(sql).all(...values) as unknown as PredictionRecord[];
}

export function getPredictionStats(): {
  totalPredictions: number;
  legitimatePredictions: number;
  fraudulentPredictions: number;
  avgProbability: number;
} {
  const database = getDb();
  const row = database
    .prepare(
      `SELECT
        COUNT(*) as total,
        SUM(CASE WHEN prediction = 'LEGITIMATE' THEN 1 ELSE 0 END) as legit,
        SUM(CASE WHEN prediction = 'FRAUDULENT' THEN 1 ELSE 0 END) as fraud,
        COALESCE(AVG(fraud_probability), 0) as avg_prob
      FROM predictions`
    )
    .get() as { total: number; legit: number | null; fraud: number | null; avg_prob: number };

  return {
    totalPredictions: Number(row.total || 0),
    legitimatePredictions: Number(row.legit || 0),
    fraudulentPredictions: Number(row.fraud || 0),
    avgProbability: Number(row.avg_prob || 0),
  };
}

export function getDatabaseStats(): {
  dbPath: string;
  totalRecords: number;
  lastCreatedAt: string | null;
} {
  const database = getDb();
  const row = database
    .prepare('SELECT COUNT(*) as total, MAX(created_at) as last_at FROM predictions')
    .get() as { total: number; last_at: string | null };

  return {
    dbPath: DB_PATH,
    totalRecords: Number(row.total || 0),
    lastCreatedAt: row.last_at ?? null,
  };
}

export function clearPredictions(): number {
  const database = getDb();
  const info = database.prepare('DELETE FROM predictions').run();
  return Number(info.changes || 0);
}
