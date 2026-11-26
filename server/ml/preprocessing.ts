import fs from 'fs';
import readline from 'readline';
import { FEATURE_NAMES } from '../config.ts';
import { ColumnProfile, DatasetSummary, EdaReport, PreprocessingArtifact } from './types.ts';

export interface ParsedDataset {
  features: Float64Array[]; // deduplicated feature vectors (length 30)
  labels: Uint8Array;       // deduplicated targets (0 or 1)
  summary: DatasetSummary;
  edaReport: EdaReport;
}

export interface StratifiedSplits {
  trainX: Float64Array[];
  trainY: Uint8Array;
  valX: Float64Array[];
  valY: Uint8Array;
  testX: Float64Array[];
  testY: Uint8Array;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleIndices(indices: number[], rand: () => number): number[] {
  const arr = indices.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function medianSorted(sorted: number[]): number {
  if (sorted.length === 0) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function quantileSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function stripQuotes(s: string): string {
  const trimmed = s.replace(/^\uFEFF/, '').trim();
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return trimmed.slice(1, -1).trim();
  }
  return trimmed;
}

function detectDelimiter(headerLine: string): string {
  const commaCount = (headerLine.match(/,/g) || []).length;
  const semiCount = (headerLine.match(/;/g) || []).length;
  const tabCount = (headerLine.match(/\t/g) || []).length;
  if (semiCount > commaCount && semiCount >= tabCount) return ';';
  if (tabCount > commaCount && tabCount > semiCount) return '\t';
  return ',';
}

function buildColumnMap(cols: string[]): Map<string, number> {
  const map = new Map<string, number>();
  cols.forEach((c, idx) => {
    map.set(c, idx);
    map.set(c.toLowerCase(), idx);
  });
  return map;
}

function lookupCol(colMap: Map<string, number>, name: string): number | undefined {
  return colMap.get(name) ?? colMap.get(name.toLowerCase());
}

/**
 * Phase 1 & Phase 2 Step 1:
 * Loads CSV, computes dataset shape, columns & dtypes, missing value counts,
 * duplicate investigation (before removal), Pearson feature correlations with Class,
 * and applies documented exact-row deduplication prior to stratified splitting.
 */
export async function parseCreditCardCsvFile(
  filePath: string,
  source: 'kaggle_csv' | 'benchmark_ulb',
  filename: string
): Promise<ParsedDataset> {
  const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  const d = FEATURE_NAMES.length;
  let delimiter = ',';
  let headerIndices: number[] | null = null;
  let classIndex = -1;
  let totalRawRows = 0;

  const missingPerCol = new Array<number>(d + 1).fill(0);
  let missingValues = 0;

  // Duplicate investigation counters
  let totalDuplicates = 0;
  let legitDuplicates = 0;
  let fraudDuplicates = 0;
  const seenExactRows = new Set<string>();

  // Online Welford/moment accumulators for correlations & column stats (computed on valid rows)
  const colSum = new Float64Array(d);
  const colSqSum = new Float64Array(d);
  const colCrossClass = new Float64Array(d);
  const colMin = new Float64Array(d).fill(Infinity);
  const colMax = new Float64Array(d).fill(-Infinity);
  let validCountForCorr = 0;
  let classSum = 0;

  const features: Float64Array[] = [];
  const labelList: number[] = [];

  const legitAmounts: number[] = [];
  const fraudAmounts: number[] = [];
  let legitSum = 0;
  let fraudSum = 0;
  let legitMax = 0;
  let fraudMax = 0;

  const bucketDefs = [
    { label: '$0–10', max: 10 },
    { label: '$10–50', max: 50 },
    { label: '$50–100', max: 100 },
    { label: '$100–250', max: 250 },
    { label: '$250–500', max: 500 },
    { label: '$500–1k', max: 1000 },
    { label: '$1k+', max: Infinity },
  ];
  const legitBuckets = new Array(bucketDefs.length).fill(0);
  const fraudBuckets = new Array(bucketDefs.length).fill(0);

  const hourStats = new Map<number, { total: number; fraud: number; amountSum: number }>();
  const highRiskCandidates: Array<{
    rowIndex: number;
    time: number;
    amount: number;
    v14: number;
    v17: number;
    v12: number;
    classLabel: number;
  }> = [];

  let lineNumber = 0;
  for await (const rawLine of rl) {
    const line = rawLine.replace(/^\uFEFF/, '').trim();
    if (!line) continue;
    lineNumber++;

    if (lineNumber === 1) {
      delimiter = detectDelimiter(line);
      const cols = line.split(delimiter).map(stripQuotes);
      const colMap = buildColumnMap(cols);

      const missingCols = FEATURE_NAMES.filter((f) => lookupCol(colMap, f) === undefined);
      if (missingCols.length > 0) {
        throw new Error(
          `CSV is missing required feature columns (${missingCols.slice(0, 5).join(', ')}). Found columns: ${cols.slice(0, 8).join(', ')}. Expected ULB schema: Time, V1..V28, Amount, Class.`
        );
      }
      const clsIdx = lookupCol(colMap, 'Class');
      if (clsIdx === undefined) {
        throw new Error(
          'Training CSV must contain the target "Class" column (0 = Legitimate, 1 = Fraudulent). For scoring unlabelled transactions, use Fraud Detection -> Batch CSV Scoring.'
        );
      }

      headerIndices = FEATURE_NAMES.map((f) => lookupCol(colMap, f)!);
      classIndex = clsIdx;
      continue;
    }

    if (!headerIndices) continue;
    totalRawRows++;
    const parts = line.split(delimiter);
    if (parts.length < d + 1) {
      missingValues++;
      continue;
    }

    const row = new Float64Array(d);
    let invalid = false;
    for (let j = 0; j < d; j++) {
      const rawToken = stripQuotes(parts[headerIndices[j]] ?? '');
      const val = rawToken === '' ? NaN : Number(rawToken);
      if (!Number.isFinite(val)) {
        missingPerCol[j]++;
        invalid = true;
      } else {
        row[j] = val;
      }
    }

    const clsToken = stripQuotes(parts[classIndex] ?? '');
    const clsRaw = clsToken === '' ? NaN : Number(clsToken);
    if (!Number.isFinite(clsRaw) || (clsRaw !== 0 && clsRaw !== 1)) {
      missingPerCol[d]++;
      invalid = true;
    }

    if (invalid) {
      missingValues++;
      continue;
    }

    // Exact full-row duplicate check across Time, V1..V28, Amount, Class
    const exactSig = `${row[0]}|${row[1].toFixed(6)}|${row[4].toFixed(6)}|${row[12].toFixed(6)}|${row[14].toFixed(6)}|${row[17].toFixed(6)}|${row[29].toFixed(2)}|${clsRaw}`;
    if (seenExactRows.has(exactSig)) {
      totalDuplicates++;
      if (clsRaw === 1) fraudDuplicates++;
      else legitDuplicates++;
      // Documented decision: skip exact duplicate rows so identical records never leak across train/test splits
      continue;
    }
    if (seenExactRows.size < 350000) {
      seenExactRows.add(exactSig);
    }

    // Update EDA summary & correlation accumulators
    validCountForCorr++;
    classSum += clsRaw;
    for (let j = 0; j < d; j++) {
      const val = row[j];
      colSum[j] += val;
      colSqSum[j] += val * val;
      if (clsRaw === 1) colCrossClass[j] += val;
      if (val < colMin[j]) colMin[j] = val;
      if (val > colMax[j]) colMax[j] = val;
    }

    const rowIdx = features.length;
    features.push(row);
    labelList.push(clsRaw);

    const txTime = row[0];
    const amount = Math.max(0, row[29]);

    const bIdx = bucketDefs.findIndex((b) => amount < b.max);
    const resolvedBucket = bIdx === -1 ? bucketDefs.length - 1 : bIdx;

    if (clsRaw === 1) {
      fraudAmounts.push(amount);
      fraudSum += amount;
      if (amount > fraudMax) fraudMax = amount;
      fraudBuckets[resolvedBucket]++;
      if (highRiskCandidates.length < 30) {
        highRiskCandidates.push({
          rowIndex: rowIdx + 1,
          time: txTime,
          amount,
          v14: Number(row[14].toFixed(4)),
          v17: Number(row[17].toFixed(4)),
          v12: Number(row[12].toFixed(4)),
          classLabel: clsRaw,
        });
      }
    } else {
      legitAmounts.push(amount);
      legitSum += amount;
      if (amount > legitMax) legitMax = amount;
      legitBuckets[resolvedBucket]++;
    }

    const hour = Math.max(0, Math.min(47, Math.floor(txTime / 3600)));
    const curHour = hourStats.get(hour) || { total: 0, fraud: 0, amountSum: 0 };
    curHour.total++;
    if (clsRaw === 1) curHour.fraud++;
    curHour.amountSum += amount;
    hourStats.set(hour, curHour);
  }

  if (features.length < 50) {
    throw new Error('Dataset contains too few valid rows (minimum 50 rows required).');
  }

  const fraudCount = fraudAmounts.length;
  const legitimateCount = legitAmounts.length;
  if (fraudCount < 5 || legitimateCount < 20) {
    throw new Error('Dataset requires at least 5 fraudulent and 20 legitimate transactions for stratified splitting.');
  }

  legitAmounts.sort((a, b) => a - b);
  fraudAmounts.sort((a, b) => a - b);

  // Compute Pearson correlation with Class and per-column profile
  const n = validCountForCorr;
  const classMean = classSum / n;
  const classVar = Math.max(1e-12, classSum - n * classMean * classMean);
  const columns: ColumnProfile[] = [];

  for (let j = 0; j < d; j++) {
    const mean = colSum[j] / n;
    const varSum = Math.max(0, colSqSum[j] - n * mean * mean);
    const std = Math.sqrt(varSum / Math.max(1, n - 1));
    const covSum = colCrossClass[j] - n * mean * classMean;
    const denom = Math.sqrt(varSum * classVar);
    const corr = denom > 1e-12 ? covSum / denom : 0;

    columns.push({
      name: FEATURE_NAMES[j],
      dtype: 'float64',
      missingCount: missingPerCol[j],
      mean: Number(mean.toFixed(4)),
      std: Number(std.toFixed(4)),
      min: Number(colMin[j].toFixed(4)),
      max: Number(colMax[j].toFixed(4)),
      correlationWithClass: Number(corr.toFixed(4)),
    });
  }

  columns.push({
    name: 'Class',
    dtype: 'int64',
    missingCount: missingPerCol[d],
    mean: Number(classMean.toFixed(6)),
    std: Number(Math.sqrt(classVar / Math.max(1, n - 1)).toFixed(4)),
    min: 0,
    max: 1,
    correlationWithClass: 1,
  });

  const topCorrelations = columns
    .filter((c) => c.name !== 'Class')
    .map((c) => ({
      feature: c.name,
      correlation: c.correlationWithClass,
      direction: (c.correlationWithClass >= 0 ? 'positive' : 'negative') as 'positive' | 'negative',
    }))
    .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
    .slice(0, 12);

  const sortedHours = Array.from(hourStats.keys()).sort((a, b) => a - b);
  const timeTrends = sortedHours.map((h) => {
    const entry = hourStats.get(h)!;
    return {
      hourBucket: `${h}h`,
      total: entry.total,
      fraudulent: entry.fraud,
      fraudRate: entry.total > 0 ? Number(((entry.fraud / entry.total) * 100).toFixed(3)) : 0,
      avgAmount: entry.total > 0 ? Number((entry.amountSum / entry.total).toFixed(2)) : 0,
    };
  });

  highRiskCandidates.sort(
    (a, b) => Math.abs(b.v14) + Math.abs(b.v17) - (Math.abs(a.v14) + Math.abs(a.v17))
  );

  const nowIso = new Date().toISOString();
  const labels = Uint8Array.from(labelList);

  const summary: DatasetSummary = {
    loaded: true,
    source,
    filename,
    totalRows: features.length,
    legitimateCount,
    fraudCount,
    fraudPercentage: Number(((fraudCount / features.length) * 100).toFixed(4)),
    featureCount: FEATURE_NAMES.length,
    missingValues,
    duplicateRows: totalDuplicates,
    amountStats: {
      legitMean: Number((legitSum / legitimateCount).toFixed(2)),
      legitMedian: Number(medianSorted(legitAmounts).toFixed(2)),
      legitMax: Number(legitMax.toFixed(2)),
      fraudMean: Number((fraudSum / fraudCount).toFixed(2)),
      fraudMedian: Number(medianSorted(fraudAmounts).toFixed(2)),
      fraudMax: Number(fraudMax.toFixed(2)),
    },
    amountHistogram: bucketDefs.map((b, idx) => ({
      bucket: b.label,
      legitimate: legitBuckets[idx],
      fraudulent: fraudBuckets[idx],
    })),
    timeTrends,
    highRiskSamples: highRiskCandidates.slice(0, 12),
    loadedAt: nowIso,
  };

  const edaReport: EdaReport = {
    generated: true,
    source,
    filename,
    shape: { rows: totalRawRows, columns: d + 1 },
    columns,
    missingValuesTotal: missingValues,
    duplicateInvestigation: {
      totalDuplicates,
      legitDuplicates,
      fraudDuplicates,
      duplicateRatePct: Number(((totalDuplicates / Math.max(1, totalRawRows)) * 100).toFixed(3)),
      rowsAfterDeduplication: features.length,
      handlingDecision:
        'Exact duplicate transactions were investigated and removed prior to stratified splitting to prevent identical records from appearing in both training and held-out test folds (data leakage prevention).',
    },
    topCorrelations,
    generatedAt: nowIso,
  };

  return { features, labels, summary, edaReport };
}

export function parseInferenceBatchCsv(csvText: string): {
  rows: Array<{ reference?: string; features: Float64Array }>;
  invalidCount: number;
} {
  const lines = csvText
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    throw new Error('CSV must include a header row and at least one transaction row.');
  }

  const delimiter = detectDelimiter(lines[0]);
  const cols = lines[0].split(delimiter).map(stripQuotes);
  const colMap = buildColumnMap(cols);

  const missingCols = FEATURE_NAMES.filter((f) => lookupCol(colMap, f) === undefined);
  if (missingCols.length > 0) {
    throw new Error(
      `Batch CSV missing required columns (${missingCols.slice(0, 5).join(', ')}). Found: ${cols.slice(0, 8).join(', ')}.`
    );
  }

  const headerIndices = FEATURE_NAMES.map((f) => lookupCol(colMap, f)!);
  const refIndex = lookupCol(colMap, 'transaction_reference') ?? lookupCol(colMap, 'reference') ?? -1;

  const rows: Array<{ reference?: string; features: Float64Array }> = [];
  let invalidCount = 0;

  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(delimiter);
    const vec = new Float64Array(FEATURE_NAMES.length);
    let invalid = false;

    for (let j = 0; j < FEATURE_NAMES.length; j++) {
      const val = Number(stripQuotes(parts[headerIndices[j]] ?? ''));
      if (!Number.isFinite(val)) {
        invalid = true;
        break;
      }
      vec[j] = val;
    }

    if (invalid) {
      invalidCount++;
      continue;
    }

    const reference = refIndex >= 0 ? stripQuotes(parts[refIndex] ?? '') : undefined;
    rows.push({
      reference: reference && reference.length > 0 ? reference : undefined,
      features: vec,
    });
  }

  if (rows.length === 0) {
    throw new Error('No valid numeric transaction rows found in uploaded CSV.');
  }

  return { rows, invalidCount };
}

export function stratifiedSplit(
  features: Float64Array[],
  labels: Uint8Array,
  trainRatio = 0.70,
  valRatio = 0.15
): StratifiedSplits {
  const rand = mulberry32(42);
  const posIndices: number[] = [];
  const negIndices: number[] = [];

  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) posIndices.push(i);
    else negIndices.push(i);
  }

  const posShuffled = shuffleIndices(posIndices, rand);
  const negShuffled = shuffleIndices(negIndices, rand);

  const posTrainEnd = Math.max(2, Math.floor(posShuffled.length * trainRatio));
  const posValEnd = Math.max(posTrainEnd + 1, Math.floor(posShuffled.length * (trainRatio + valRatio)));

  const negTrainEnd = Math.max(10, Math.floor(negShuffled.length * trainRatio));
  const negValEnd = Math.max(negTrainEnd + 2, Math.floor(negShuffled.length * (trainRatio + valRatio)));

  const trainIdx = shuffleIndices(
    posShuffled.slice(0, posTrainEnd).concat(negShuffled.slice(0, negTrainEnd)),
    rand
  );
  const valIdx = shuffleIndices(
    posShuffled.slice(posTrainEnd, posValEnd).concat(negShuffled.slice(negTrainEnd, negValEnd)),
    rand
  );
  const testIdx = shuffleIndices(
    posShuffled.slice(posValEnd).concat(negShuffled.slice(negValEnd)),
    rand
  );

  const gather = (idxs: number[]) => {
    const x: Float64Array[] = new Array(idxs.length);
    const y = new Uint8Array(idxs.length);
    for (let i = 0; i < idxs.length; i++) {
      x[i] = features[idxs[i]];
      y[i] = labels[idxs[i]];
    }
    return { x, y };
  };

  const train = gather(trainIdx);
  const val = gather(valIdx);
  const test = gather(testIdx);

  return {
    trainX: train.x,
    trainY: train.y,
    valX: val.x,
    valY: val.y,
    testX: test.x,
    testY: test.y,
  };
}

/**
 * Phase 2: Fits RobustScaler (median / IQR) ONLY on Time (index 0) and Amount (index 29)
 * strictly using training data, while preserving existing V1–V28 PCA-transformed features.
 */
export function fitPreprocessing(
  trainX: Float64Array[],
  trainY: Uint8Array,
  valY: Uint8Array,
  testY: Uint8Array,
  duplicatesRemoved = 0
): PreprocessingArtifact {
  const n = trainX.length;

  const times = new Array<number>(n);
  const amounts = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    times[i] = trainX[i][0];
    amounts[i] = trainX[i][29];
  }
  times.sort((a, b) => a - b);
  amounts.sort((a, b) => a - b);

  const timeMedian = medianSorted(times);
  const timeIqrRaw = quantileSorted(times, 0.75) - quantileSorted(times, 0.25);
  const timeIqr = timeIqrRaw > 1e-6 ? timeIqrRaw : 1;

  const amountMedian = medianSorted(amounts);
  const amountIqrRaw = quantileSorted(amounts, 0.75) - quantileSorted(amounts, 0.25);
  const amountIqr = amountIqrRaw > 1e-6 ? amountIqrRaw : 1;

  let trainFraudCount = 0;
  for (let i = 0; i < trainY.length; i++) if (trainY[i] === 1) trainFraudCount++;
  let valFraudCount = 0;
  for (let i = 0; i < valY.length; i++) if (valY[i] === 1) valFraudCount++;
  let testFraudCount = 0;
  for (let i = 0; i < testY.length; i++) if (testY[i] === 1) testFraudCount++;

  return {
    featureNames: [...FEATURE_NAMES],
    timeMedian,
    timeIqr,
    amountMedian,
    amountIqr,
    vFeaturesPreserved: true,
    duplicatesRemoved,
    trainRows: trainX.length,
    valRows: valY.length,
    testRows: testY.length,
    trainFraudCount,
    valFraudCount,
    testFraudCount,
    fittedAt: new Date().toISOString(),
  };
}

export function transformRow(raw: Float64Array | number[], artifact: PreprocessingArtifact): Float64Array {
  const d = FEATURE_NAMES.length;
  const out = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    if (j === 0) {
      out[j] = (raw[0] - artifact.timeMedian) / artifact.timeIqr;
    } else if (j === 29) {
      out[j] = (raw[29] - artifact.amountMedian) / artifact.amountIqr;
    } else {
      // Preserve existing V1–V28 PCA-transformed features intact
      out[j] = raw[j];
    }
  }
  return out;
}

export function transformDataset(rows: Float64Array[], artifact: PreprocessingArtifact): Float64Array[] {
  const out = new Array<Float64Array>(rows.length);
  for (let i = 0; i < rows.length; i++) {
    out[i] = transformRow(rows[i], artifact);
  }
  return out;
}

function gaussianRandom(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Generates an explicitly labeled ULB-schema benchmark CSV (Time, V1..V28, Amount, Class)
 * for immediate end-to-end ML training & evaluation when the user does not yet have
 * the 150MB Kaggle creditcard.csv file on disk.
 */
export function writeBenchmarkUlbCsv(targetPath: string, totalRows = 6000): void {
  const rand = mulberry32(202611);
  const header = [...FEATURE_NAMES, 'Class'].join(',');
  const lines: string[] = [header];

  const fraudRate = 0.018;

  for (let i = 0; i < totalRows; i++) {
    const isFraud = rand() < fraudRate ? 1 : 0;
    const time = Math.round((i / totalRows) * 172792 + rand() * 30);

    const v = new Array<number>(28);
    for (let k = 0; k < 28; k++) {
      v[k] = gaussianRandom(rand) * (0.95 - k * 0.015);
    }

    let amount = 0;
    if (isFraud === 1) {
      const pattern = rand();
      if (pattern < 0.55) {
        v[13] = -4.2 + gaussianRandom(rand) * 1.8;
        v[16] = -4.6 + gaussianRandom(rand) * 2.1;
        v[11] = -3.8 + gaussianRandom(rand) * 1.6;
        v[9] = -3.1 + gaussianRandom(rand) * 1.4;
        v[3] = 2.9 + gaussianRandom(rand) * 1.3;
        v[10] = 2.7 + gaussianRandom(rand) * 1.2;
        amount = Math.exp(3.4 + gaussianRandom(rand) * 1.45);
      } else if (pattern < 0.85) {
        v[13] = -2.1 + gaussianRandom(rand) * 1.4;
        v[16] = -1.9 + gaussianRandom(rand) * 1.5;
        v[11] = -1.7 + gaussianRandom(rand) * 1.3;
        v[2] = -2.2 + gaussianRandom(rand) * 1.2;
        v[3] = 1.6 + gaussianRandom(rand) * 1.1;
        amount = Math.exp(2.2 + gaussianRandom(rand) * 1.8);
      } else {
        v[13] = -1.2 + gaussianRandom(rand) * 1.6;
        v[6] = 2.4 + gaussianRandom(rand) * 1.4;
        v[0] = -2.8 + gaussianRandom(rand) * 1.5;
        amount = 450 + Math.abs(gaussianRandom(rand)) * 680;
      }
    } else {
      if (rand() < 0.012) {
        v[13] = -1.8 + gaussianRandom(rand) * 0.8;
        v[3] = 1.5 + gaussianRandom(rand) * 0.8;
      }
      amount = Math.exp(2.8 + gaussianRandom(rand) * 1.35);
    }

    amount = Math.min(25691, Math.max(0.01, Number(amount.toFixed(2))));
    const rowVals = [
      time.toString(),
      ...v.map((val) => val.toFixed(6)),
      amount.toFixed(2),
      isFraud.toString(),
    ];
    lines.push(rowVals.join(','));
  }

  fs.writeFileSync(targetPath, lines.join('\n'), 'utf8');
}
