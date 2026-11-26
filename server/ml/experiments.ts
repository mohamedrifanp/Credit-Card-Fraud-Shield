import { FEATURE_NAMES } from '../config.ts';
import { evaluateModelOnTest } from './evaluate.ts';
import { predictLogisticRegression, trainLogisticRegression } from './models.ts';
import { mulberry32 } from './preprocessing.ts';
import {
  DBSCANExperimentResult,
  DBSCANPoint,
  ModelEvaluation,
  PCAExperimentResult,
} from './types.ts';

/**
 * Symmetric 30x30 Jacobi Eigenvalue Decomposition for PCA on standardized training features.
 */
function jacobiEigen(matrix: number[][], maxIter = 80): {
  eigenvalues: number[];
  eigenvectors: number[][];
} {
  const n = matrix.length;
  const a = matrix.map((r) => r.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let iter = 0; iter < maxIter; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(a[i][j]);
        if (val > maxVal) {
          maxVal = val;
          p = i;
          q = j;
        }
      }
    }
    if (maxVal < 1e-8) break;

    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const aip = a[i][p];
        const aiq = a[i][q];
        a[i][p] = c * aip - s * aiq;
        a[p][i] = a[i][p];
        a[i][q] = s * aip + c * aiq;
        a[q][i] = a[i][q];
      }
    }

    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = 0;
    a[q][p] = 0;

    for (let i = 0; i < n; i++) {
      const vip = v[i][p];
      const viq = v[i][q];
      v[i][p] = c * vip - s * viq;
      v[i][q] = s * vip + c * viq;
    }
  }

  const pairs = Array.from({ length: n }, (_, i) => ({
    val: Math.max(0, a[i][i]),
    vec: v.map((row) => row[i]),
  })).sort((x, y) => y.val - x.val);

  return {
    eigenvalues: pairs.map((p) => p.val),
    eigenvectors: pairs.map((p) => p.vec),
  };
}

function projectRows(
  rows: Float64Array[],
  components: number[][],
  k: number
): Float64Array[] {
  const out = new Array<Float64Array>(rows.length);
  const d = FEATURE_NAMES.length;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const proj = new Float64Array(d); // pad remaining dims with 0 so LogisticRegression interface stays unified
    for (let c = 0; c < k; c++) {
      const vec = components[c];
      let dot = 0;
      for (let j = 0; j < d; j++) dot += r[j] * vec[j];
      proj[c] = dot;
    }
    out[i] = proj;
  }
  return out;
}

export function runPCAExperiment(
  trainX: Float64Array[],
  trainY: Uint8Array,
  testX: Float64Array[],
  testY: Uint8Array,
  baselineEval: ModelEvaluation
): { pcaResult: PCAExperimentResult; eigenvectors: number[][] } {
  const d = FEATURE_NAMES.length;
  const maxSample = Math.min(trainX.length, 6000);
  const cov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));

  for (let i = 0; i < maxSample; i++) {
    const row = trainX[i];
    for (let j = 0; j < d; j++) {
      const rj = row[j];
      for (let k = j; k < d; k++) {
        cov[j][k] += rj * row[k];
      }
    }
  }
  const denom = Math.max(1, maxSample - 1);
  for (let j = 0; j < d; j++) {
    for (let k = j; k < d; k++) {
      cov[j][k] /= denom;
      cov[k][j] = cov[j][k];
    }
  }

  const { eigenvalues, eigenvectors } = jacobiEigen(cov);
  const totalVar = eigenvalues.reduce((a, b) => a + b, 0) || 1;
  const explainedVarianceRatio = eigenvalues.map((v) => Number((v / totalVar).toFixed(4)));

  const cumulativeVariance: number[] = [];
  let running = 0;
  let componentsFor95Pct = d;
  for (let i = 0; i < explainedVarianceRatio.length; i++) {
    running += explainedVarianceRatio[i];
    cumulativeVariance.push(Number(Math.min(1, running).toFixed(4)));
    if (running >= 0.95 && componentsFor95Pct === d) {
      componentsFor95Pct = i + 1;
    }
  }

  const reducedK = Math.max(2, Math.min(d - 1, componentsFor95Pct));
  const trainProj = projectRows(trainX, eigenvectors, reducedK);
  const testProj = projectRows(testX, eigenvectors, reducedK);

  const pcaModel = trainLogisticRegression(trainProj, trainY, 90, 0.002);
  const pcaProbs = predictLogisticRegression(pcaModel, testProj);
  const pcaEval = evaluateModelOnTest(
    'logistic_regression',
    testY,
    pcaProbs,
    0.5,
    pcaModel.trainingTimeMs,
    pcaModel.weights
  );

  return {
    eigenvectors,
    pcaResult: {
      explainedVarianceRatio,
      cumulativeVariance,
      componentsFor95Pct: reducedK,
      beforePca: {
        precision: baselineEval.precision,
        recall: baselineEval.recall,
        f1Score: baselineEval.f1Score,
        rocAuc: baselineEval.rocAuc,
        prAuc: baselineEval.prAuc,
        dimensions: d,
      },
      afterPca: {
        precision: pcaEval.precision,
        recall: pcaEval.recall,
        f1Score: pcaEval.f1Score,
        rocAuc: pcaEval.rocAuc,
        prAuc: pcaEval.prAuc,
        dimensions: reducedK,
      },
      note:
        'Features V1–V28 in the ULB dataset are already orthogonal principal components from the original cardholder data. Applying a second PCA is retained strictly as an experimental dimensionality-reduction benchmark rather than a mandatory preprocessing step.',
    },
  };
}

/**
 * Exploratory DBSCAN Anomaly Detection in high-dimensional scaled feature space.
 * Uses a stratified subsample (<= 1,800 rows) to keep O(N^2) Euclidean neighborhood scans
 * responsive on CPU within 8 GB RAM constraints, and treats noise points (-1) strictly
 * as structural anomalies evaluated post-hoc against fraud labels.
 */
export function runDBSCANExperiment(
  scaledX: Float64Array[],
  rawX: Float64Array[],
  labels: Uint8Array,
  eigenvectors: number[][]
): DBSCANExperimentResult {
  const rand = mulberry32(909);
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) posIdx.push(i);
    else negIdx.push(i);
  }

  // Keep up to 120 frauds and 1,480 legitimate points for fast density clustering
  const selected: number[] = [];
  for (let i = 0; i < posIdx.length && selected.length < 120; i++) {
    selected.push(posIdx[i]);
  }
  const targetNeg = Math.min(negIdx.length, 1480);
  const step = Math.max(1, Math.floor(negIdx.length / targetNeg));
  for (let i = 0; i < negIdx.length && selected.length < 1600; i += step) {
    selected.push(negIdx[i]);
  }
  // Shuffle deterministically
  for (let i = selected.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = selected[i];
    selected[i] = selected[j];
    selected[j] = t;
  }

  const n = selected.length;
  // Use top 12 PCs for DBSCAN distance to mitigate the curse of dimensionality in 30D space
  const dims = Math.min(12, eigenvectors.length);
  const pts = new Array<Float64Array>(n);
  for (let i = 0; i < n; i++) {
    const row = scaledX[selected[i]];
    const proj = new Float64Array(dims);
    for (let c = 0; c < dims; c++) {
      let dot = 0;
      const vec = eigenvectors[c];
      for (let j = 0; j < row.length; j++) dot += row[j] * vec[j];
      proj[c] = dot;
    }
    pts[i] = proj;
  }

  // Calibrate eps automatically from 85th percentile of 5-NN distances on a probe sample
  const probeSize = Math.min(n, 220);
  const kDistances: number[] = [];
  for (let i = 0; i < probeSize; i++) {
    const dists: number[] = [];
    const pi = pts[i];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const pj = pts[j];
      let sq = 0;
      for (let d = 0; d < dims; d++) {
        const diff = pi[d] - pj[d];
        sq += diff * diff;
      }
      dists.push(Math.sqrt(sq));
    }
    dists.sort((a, b) => a - b);
    kDistances.push(dists[Math.min(4, dists.length - 1)]);
  }
  kDistances.sort((a, b) => a - b);
  const eps = Number(Math.max(1.8, kDistances[Math.floor(kDistances.length * 0.88)] || 3.2).toFixed(2));
  const epsSq = eps * eps;
  const minSamples = 6;

  // 0 = unvisited, -1 = noise, >0 = clusterId
  const clusterIds = new Int32Array(n);
  let currentCluster = 0;

  const regionQuery = (idx: number): number[] => {
    const neighbors: number[] = [];
    const p = pts[idx];
    for (let j = 0; j < n; j++) {
      const q = pts[j];
      let sq = 0;
      for (let d = 0; d < dims; d++) {
        const diff = p[d] - q[d];
        sq += diff * diff;
      }
      if (sq <= epsSq) neighbors.push(j);
    }
    return neighbors;
  };

  for (let i = 0; i < n; i++) {
    if (clusterIds[i] !== 0) continue;
    const neighbors = regionQuery(i);
    if (neighbors.length < minSamples) {
      clusterIds[i] = -1;
      continue;
    }
    currentCluster++;
    clusterIds[i] = currentCluster;
    const seedSet = neighbors.filter((idx) => idx !== i);

    for (let s = 0; s < seedSet.length; s++) {
      const curr = seedSet[s];
      if (clusterIds[curr] === -1) {
        clusterIds[curr] = currentCluster;
      }
      if (clusterIds[curr] !== 0) continue;
      clusterIds[curr] = currentCluster;
      const currNeighbors = regionQuery(curr);
      if (currNeighbors.length >= minSamples) {
        for (const nb of currNeighbors) {
          if (clusterIds[nb] === 0 || clusterIds[nb] === -1) {
            seedSet.push(nb);
          }
        }
      }
    }
  }

  let noiseCount = 0;
  let noiseFraudCount = 0;
  let totalFraudInSample = 0;
  const scatterPoints: DBSCANPoint[] = [];

  for (let i = 0; i < n; i++) {
    const origIdx = selected[i];
    const actualClass = labels[origIdx];
    const isNoise = clusterIds[i] === -1;
    if (actualClass === 1) totalFraudInSample++;
    if (isNoise) {
      noiseCount++;
      if (actualClass === 1) noiseFraudCount++;
    }

    // Limit scatter visualization points to <= 700 (keeping all noise & fraud points)
    if (isNoise || actualClass === 1 || scatterPoints.length < 700) {
      scatterPoints.push({
        x: Number(pts[i][0].toFixed(3)),
        y: Number((pts[i][1] ?? 0).toFixed(3)),
        cluster: clusterIds[i],
        isNoise,
        actualClass,
        amount: Number(rawX[origIdx][29].toFixed(2)),
      });
    }
  }

  const noiseFraudRecall = totalFraudInSample > 0 ? noiseFraudCount / totalFraudInSample : 0;
  const noiseFraudPrecision = noiseCount > 0 ? noiseFraudCount / noiseCount : 0;

  return {
    eps,
    minSamples,
    sampledCount: n,
    clusterCount: currentCluster,
    noiseCount,
    noisePercentage: Number(((noiseCount / n) * 100).toFixed(2)),
    noiseFraudCount,
    totalFraudInSample,
    noiseFraudRecall: Number(noiseFraudRecall.toFixed(4)),
    noiseFraudPrecision: Number(noiseFraudPrecision.toFixed(4)),
    points: scatterPoints.slice(0, 750),
    note:
      'DBSCAN projects scaled features onto the top 12 principal components to mitigate high-dimensional distance concentration. Noise points (Cluster -1) indicate structural density anomalies and are compared against Class labels purely for post-hoc evaluation—never as confirmed fraud.',
  };
}
