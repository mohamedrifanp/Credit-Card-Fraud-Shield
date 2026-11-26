import crypto from 'crypto';
import express, { NextFunction, Request, Response } from 'express';
import fs from 'fs';
import multer from 'multer';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import {
  COMPARISON_CSV_PATH,
  DATASET_PATH,
  DATA_DIR,
  ensureDirectories,
  FEATURE_NAMES,
  MODEL_IDS,
  ModelId,
} from './server/config.ts';
import {
  clearPredictions,
  getDatabaseStats,
  getPredictionStats,
  getSettings,
  queryPredictions,
  updateSettings,
} from './server/database.ts';
import {
  getPipelineStatus,
  initializePipeline,
  inspectDatasetEda,
  predictBatchTransactions,
  predictSingleTransaction,
  runFullTrainingPipeline,
} from './server/ml/pipeline.ts';
import { parseInferenceBatchCsv } from './server/ml/preprocessing.ts';

ensureDirectories();
initializePipeline();

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '10mb' }));
app.use(express.text({ limit: '10mb', type: 'text/csv' }));

// Safe multipart upload confined to DATA_DIR
const upload = multer({
  dest: DATA_DIR,
  limits: {
    fileSize: 220 * 1024 * 1024, // Supports full ~150MB Kaggle creditcard.csv
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext !== '.csv') {
      cb(new Error('Only .csv files are accepted.'));
      return;
    }
    cb(null, true);
  },
});

function safeError(res: Response, status: number, fallback: string, err?: unknown): void {
  const msg = err instanceof Error ? err.message : fallback;
  // Return clean operational validation errors without internal stack traces
  const sanitized = msg.length > 220 ? fallback : msg;
  res.status(status).json({ error: sanitized });
}

// ---------------------------------------------------------------------------
// API ROUTES
// ---------------------------------------------------------------------------

app.get('/api/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/api/status', (_req: Request, res: Response) => {
  try {
    const pipeline = getPipelineStatus();
    const settings = getSettings();
    const predictionStats = getPredictionStats();
    const databaseStats = getDatabaseStats();
    const recentPredictions = queryPredictions({ limit: 8 });

    res.json({
      ...pipeline,
      settings,
      predictionStats,
      databaseStats,
      recentPredictions,
    });
  } catch (err) {
    safeError(res, 500, 'Failed to retrieve application status.', err);
  }
});

// Phase 1 standalone EDA inspection
app.post('/api/eda/inspect', async (req: Request, res: Response) => {
  try {
    const mode = req.body?.mode === 'benchmark' ? 'benchmark' : 'kaggle';
    const result = await inspectDatasetEda({
      useKaggleIfPresent: mode === 'kaggle',
      generateBenchmark: mode === 'benchmark',
    });
    res.json(result);
  } catch (err) {
    safeError(res, 400, 'EDA inspection failed.', err);
  }
});

app.get('/api/reports/eda', (_req: Request, res: Response) => {
  const { edaReport } = getPipelineStatus();
  if (!edaReport.generated) {
    res.status(404).json({ error: 'EDA report has not been generated yet.' });
    return;
  }
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Content-Disposition', 'attachment; filename="eda_report.json"');
  res.send(JSON.stringify(edaReport, null, 2));
});

// Train models on existing Kaggle dataset or explicitly generated ULB benchmark dataset
app.post('/api/train', async (req: Request, res: Response) => {
  try {
    const mode = req.body?.mode === 'benchmark' ? 'benchmark' : 'kaggle';
    const result = await runFullTrainingPipeline({
      useKaggleIfPresent: mode === 'kaggle',
      generateBenchmark: mode === 'benchmark',
    });
    res.json({
      message: 'ML training and evaluation pipeline completed successfully.',
      ...result,
    });
  } catch (err) {
    safeError(res, 400, 'Model training failed.', err);
  }
});

// Chunked upload endpoints (handles large 150MB Kaggle creditcard.csv cleanly under Cloud Run's 32MB request cap)
const uploadSessions = new Map<string, { tempPath: string; filename: string; bytesReceived: number }>();

app.post('/api/dataset/upload/init', (req: Request, res: Response) => {
  try {
    const filename = typeof req.body?.filename === 'string' ? req.body.filename : 'creditcard.csv';
    if (!filename.toLowerCase().endsWith('.csv')) {
      res.status(400).json({ error: 'Only .csv files are accepted.' });
      return;
    }
    const uploadId = crypto.randomBytes(12).toString('hex');
    const tempPath = path.resolve(DATA_DIR, `.upload_${uploadId}.csv`);
    if (!tempPath.startsWith(path.resolve(DATA_DIR) + path.sep)) {
      res.status(400).json({ error: 'Invalid upload path.' });
      return;
    }
    fs.writeFileSync(tempPath, '');
    uploadSessions.set(uploadId, { tempPath, filename, bytesReceived: 0 });
    res.json({ uploadId });
  } catch (err) {
    safeError(res, 500, 'Failed to initialize CSV upload.', err);
  }
});

app.post(
  '/api/dataset/upload/chunk',
  express.raw({ type: 'application/octet-stream', limit: '10mb' }),
  (req: Request, res: Response) => {
    try {
      const uploadId = String(req.headers['x-upload-id'] || '');
      const session = uploadSessions.get(uploadId);
      if (!session) {
        res.status(400).json({ error: 'Upload session expired or not found.' });
        return;
      }
      if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
        res.status(400).json({ error: 'Empty chunk received.' });
        return;
      }
      if (session.bytesReceived + req.body.length > 250 * 1024 * 1024) {
        fs.unlinkSync(session.tempPath);
        uploadSessions.delete(uploadId);
        res.status(400).json({ error: 'CSV file exceeds 250 MB limit.' });
        return;
      }
      fs.appendFileSync(session.tempPath, req.body);
      session.bytesReceived += req.body.length;
      res.json({ bytesReceived: session.bytesReceived });
    } catch (err) {
      safeError(res, 500, 'Failed to write upload chunk.', err);
    }
  }
);

app.post('/api/dataset/upload/complete', async (req: Request, res: Response) => {
  const uploadId = String(req.body?.uploadId || '');
  const session = uploadSessions.get(uploadId);
  if (!session) {
    res.status(400).json({ error: 'Upload session not found.' });
    return;
  }
  uploadSessions.delete(uploadId);

  try {
    fs.renameSync(session.tempPath, DATASET_PATH);
    const result = await runFullTrainingPipeline({
      uploadedFilePath: DATASET_PATH,
      uploadedFilename: session.filename || 'creditcard.csv',
    });
    res.json({
      message: 'Dataset uploaded and all models trained successfully.',
      ...result,
    });
  } catch (err) {
    if (fs.existsSync(session.tempPath)) {
      fs.unlinkSync(session.tempPath);
    }
    safeError(res, 400, 'Failed to process uploaded training CSV.', err);
  }
});

// Single-shot upload for smaller CSVs
app.post('/api/dataset/upload', upload.single('file'), async (req: Request, res: Response) => {
  const uploaded = req.file;
  if (!uploaded) {
    res.status(400).json({ error: 'No CSV file was uploaded.' });
    return;
  }

  const resolvedUploaded = path.resolve(uploaded.path);
  const resolvedDataDir = path.resolve(DATA_DIR);
  if (!resolvedUploaded.startsWith(resolvedDataDir + path.sep)) {
    res.status(400).json({ error: 'Invalid upload path.' });
    return;
  }

  try {
    // Move uploaded file safely to data/creditcard.csv
    fs.renameSync(resolvedUploaded, DATASET_PATH);
    const result = await runFullTrainingPipeline({
      uploadedFilePath: DATASET_PATH,
      uploadedFilename: uploaded.originalname || 'creditcard.csv',
    });
    res.json({
      message: 'Dataset uploaded and all models trained successfully.',
      ...result,
    });
  } catch (err) {
    if (fs.existsSync(resolvedUploaded)) {
      fs.unlinkSync(resolvedUploaded);
    }
    safeError(res, 400, 'Failed to process uploaded training CSV.', err);
  }
});

// Single transaction prediction
app.post('/api/predict', (req: Request, res: Response) => {
  try {
    const body = req.body;
    if (!body || typeof body !== 'object') {
      res.status(400).json({ error: 'Request body must be a JSON object.' });
      return;
    }

    const rawFeatures = body.features;
    if (!rawFeatures || typeof rawFeatures !== 'object') {
      res.status(400).json({ error: 'Missing transaction "features" object.' });
      return;
    }

    const vector = new Float64Array(FEATURE_NAMES.length);
    for (let i = 0; i < FEATURE_NAMES.length; i++) {
      const name = FEATURE_NAMES[i];
      const val = Number(rawFeatures[name]);
      if (!Number.isFinite(val)) {
        res.status(400).json({ error: `Feature "${name}" must be a valid finite number.` });
        return;
      }
      if ((name === 'Amount' || name === 'Time') && val < 0) {
        res.status(400).json({ error: `Feature "${name}" cannot be negative.` });
        return;
      }
      vector[i] = val;
    }

    let modelId: ModelId | undefined;
    if (body.modelId !== undefined) {
      if (!(MODEL_IDS as readonly string[]).includes(String(body.modelId))) {
        res.status(400).json({ error: 'Invalid modelId specified.' });
        return;
      }
      modelId = body.modelId as ModelId;
    }

    let threshold: number | undefined;
    if (body.threshold !== undefined) {
      const th = Number(body.threshold);
      if (!Number.isFinite(th) || th < 0.05 || th > 0.95) {
        res.status(400).json({ error: 'Threshold must be between 0.05 and 0.95.' });
        return;
      }
      threshold = th;
    }

    const transactionReference =
      typeof body.transactionReference === 'string'
        ? body.transactionReference.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 48)
        : undefined;

    const output = predictSingleTransaction(vector, {
      modelId,
      threshold,
      transactionReference,
    });

    res.json(output);
  } catch (err) {
    safeError(res, 400, 'Prediction failed.', err);
  }
});

// Batch CSV prediction (multipart file or raw CSV text)
app.post('/api/predict/batch', upload.single('file'), (req: Request, res: Response) => {
  const uploaded = req.file;
  try {
    let csvText = '';
    if (uploaded) {
      const resolvedPath = path.resolve(uploaded.path);
      const resolvedDataDir = path.resolve(DATA_DIR);
      if (!resolvedPath.startsWith(resolvedDataDir + path.sep)) {
        res.status(400).json({ error: 'Invalid file path.' });
        return;
      }
      csvText = fs.readFileSync(resolvedPath, 'utf8');
      fs.unlinkSync(resolvedPath);
    } else if (typeof req.body?.csvText === 'string') {
      csvText = req.body.csvText;
    } else {
      res.status(400).json({ error: 'Provide a CSV file or csvText for batch prediction.' });
      return;
    }

    const parsed = parseInferenceBatchCsv(csvText);
    if (parsed.rows.length > 5000) {
      res.status(400).json({
        error: 'Batch prediction supports up to 5,000 rows per request. To train on the full 284,807-row creditcard.csv, upload it on Dashboard or Settings.',
      });
      return;
    }

    const rawModelId = uploaded ? req.body?.modelId : req.body?.modelId;
    let modelId: ModelId | undefined;
    if (rawModelId && (MODEL_IDS as readonly string[]).includes(String(rawModelId))) {
      modelId = rawModelId as ModelId;
    }

    let threshold: number | undefined;
    if (req.body?.threshold !== undefined) {
      const th = Number(req.body.threshold);
      if (Number.isFinite(th) && th >= 0.05 && th <= 0.95) {
        threshold = th;
      }
    }

    const batchOut = predictBatchTransactions(parsed.rows, { modelId, threshold });
    res.json({
      ...batchOut,
      invalidRowsSkipped: parsed.invalidCount,
    });
  } catch (err) {
    if (uploaded && fs.existsSync(uploaded.path)) {
      fs.unlinkSync(uploaded.path);
    }
    safeError(res, 400, 'Batch CSV prediction failed.', err);
  }
});

// Transaction History Query & Export
app.get('/api/history', (req: Request, res: Response) => {
  try {
    const search = typeof req.query.search === 'string' ? req.query.search : undefined;
    const statusRaw = typeof req.query.status === 'string' ? req.query.status : 'ALL';
    const status =
      statusRaw === 'LEGITIMATE' || statusRaw === 'FRAUDULENT' ? statusRaw : 'ALL';
    const dateFrom = typeof req.query.dateFrom === 'string' ? req.query.dateFrom : undefined;
    const dateTo = typeof req.query.dateTo === 'string' ? req.query.dateTo : undefined;
    const sortByRaw = typeof req.query.sortBy === 'string' ? req.query.sortBy : 'created_at';
    const sortBy =
      sortByRaw === 'amount' || sortByRaw === 'fraud_probability' ? sortByRaw : 'created_at';
    const sortDir = req.query.sortDir === 'asc' ? 'asc' : 'desc';
    const limit = Number(req.query.limit || 250);

    const records = queryPredictions({
      search,
      status,
      dateFrom,
      dateTo,
      sortBy,
      sortDir,
      limit,
    });
    res.json({ records, count: records.length });
  } catch (err) {
    safeError(res, 500, 'Failed to query transaction history.', err);
  }
});

app.get('/api/history/export', (req: Request, res: Response) => {
  try {
    const statusRaw = typeof req.query.status === 'string' ? req.query.status : 'ALL';
    const status =
      statusRaw === 'LEGITIMATE' || statusRaw === 'FRAUDULENT' ? statusRaw : 'ALL';
    const records = queryPredictions({ status, limit: 1000 });

    const header = [
      'id',
      'transaction_reference',
      'prediction',
      'fraud_probability',
      'model_name',
      'risk_level',
      'amount',
      'tx_time',
      'created_at',
    ].join(',');

    const lines = records.map((r) =>
      [
        r.id,
        `"${r.transaction_reference.replace(/"/g, '""')}"`,
        r.prediction,
        r.fraud_probability,
        `"${r.model_name.replace(/"/g, '""')}"`,
        r.risk_level,
        r.amount,
        r.tx_time,
        r.created_at,
      ].join(',')
    );

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="fraud_shield_history.csv"');
    res.send([header, ...lines].join('\n'));
  } catch (err) {
    safeError(res, 500, 'Failed to export history CSV.', err);
  }
});

app.delete('/api/history', (_req: Request, res: Response) => {
  try {
    const deleted = clearPredictions();
    res.json({ deleted });
  } catch (err) {
    safeError(res, 500, 'Failed to clear transaction history.', err);
  }
});

// Download model comparison report CSV
app.get('/api/reports/comparison', (_req: Request, res: Response) => {
  if (!fs.existsSync(COMPARISON_CSV_PATH)) {
    res.status(404).json({ error: 'Model comparison report not generated yet.' });
    return;
  }
  res.setHeader('Content-Type', 'text/csv; charset=utf-8');
  res.setHeader('Content-Disposition', 'attachment; filename="model_comparison.csv"');
  res.send(fs.readFileSync(COMPARISON_CSV_PATH, 'utf8'));
});

// Settings endpoints
app.get('/api/settings', (_req: Request, res: Response) => {
  try {
    res.json(getSettings());
  } catch (err) {
    safeError(res, 500, 'Failed to read settings.', err);
  }
});

app.post('/api/settings', (req: Request, res: Response) => {
  try {
    const activeModel = req.body?.activeModel as ModelId;
    const threshold = Number(req.body?.threshold);
    const updated = updateSettings(activeModel, threshold);
    res.json(updated);
  } catch (err) {
    safeError(res, 400, 'Invalid settings payload.', err);
  }
});

// Ensure Express/Multer errors on /api/* always return JSON instead of HTML
app.use('/api', (err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  safeError(res, 400, 'API request error.', err);
});

// ---------------------------------------------------------------------------
// VITE MIDDLEWARE (DEV) OR STATIC DIST (PROD)
// ---------------------------------------------------------------------------
async function startServer(): Promise<void> {
  if (process.env.NODE_ENV === 'production') {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Credit Card Fraud Shield server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
