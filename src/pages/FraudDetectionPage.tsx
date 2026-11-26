import React, { useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  Loader2,
  ShieldAlert,
  Upload,
} from 'lucide-react';
import { AppStatusResponse, ModelId, SinglePredictionOutput } from '../types';

const FEATURE_KEYS = [
  'Time',
  'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
  'Amount',
] as const;

function defaultFeatureValues(): Record<string, string> {
  const obj: Record<string, string> = {};
  for (const k of FEATURE_KEYS) obj[k] = '0';
  obj.Time = '42500';
  obj.Amount = '64.50';
  return obj;
}

function legitimatePreset(): Record<string, string> {
  const obj = defaultFeatureValues();
  obj.Time = '61240';
  obj.V1 = '1.1918';
  obj.V2 = '0.2661';
  obj.V3 = '0.1664';
  obj.V4 = '0.4481';
  obj.V10 = '-0.0890';
  obj.V12 = '0.1240';
  obj.V14 = '-0.1437';
  obj.V17 = '0.1148';
  obj.Amount = '38.40';
  return obj;
}

function suspiciousPreset(): Record<string, string> {
  const obj = defaultFeatureValues();
  obj.Time = '112400';
  obj.V1 = '-2.8450';
  obj.V2 = '2.4120';
  obj.V3 = '-3.9200';
  obj.V4 = '3.6500';
  obj.V10 = '-4.1200';
  obj.V11 = '3.3400';
  obj.V12 = '-5.2100';
  obj.V14 = '-6.1800';
  obj.V16 = '-3.4500';
  obj.V17 = '-6.4200';
  obj.Amount = '489.90';
  return obj;
}

export function FraudDetectionPage({
  status,
  onPredictionComplete,
}: {
  status: AppStatusResponse;
  onPredictionComplete: () => Promise<void>;
}) {
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [features, setFeatures] = useState<Record<string, string>>(legitimatePreset);
  const [reference, setReference] = useState('');
  const [selectedModel, setSelectedModel] = useState<ModelId>(status.settings.activeModel);
  const [threshold, setThreshold] = useState<number>(status.settings.threshold);

  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<SinglePredictionOutput | null>(null);

  const [batchResults, setBatchResults] = useState<{
    results: SinglePredictionOutput[];
    summary: { total: number; fraudulent: number; legitimate: number; deduplicated: number };
    invalidRowsSkipped: number;
  } | null>(null);
  const batchFileRef = useRef<HTMLInputElement>(null);

  const modelsTrained = status.evaluationReport.trained;

  const validationError = useMemo(() => {
    for (const k of FEATURE_KEYS) {
      const n = Number(features[k]);
      if (!Number.isFinite(n)) return `Feature ${k} must be a valid number.`;
      if ((k === 'Amount' || k === 'Time') && n < 0) {
        return `${k} cannot be negative.`;
      }
    }
    return null;
  }, [features]);

  const handlePredictSingle = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelsTrained) {
      setErrorMsg('Train the ML models on the Dashboard or Settings page before scoring.');
      return;
    }
    if (validationError) {
      setErrorMsg(validationError);
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    try {
      const numericFeatures: Record<string, number> = {};
      for (const k of FEATURE_KEYS) numericFeatures[k] = Number(features[k]);

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: numericFeatures,
          modelId: selectedModel,
          threshold,
          transactionReference: reference.trim() || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction request failed.');
      setSingleResult(data as SinglePredictionOutput);
      await onPredictionComplete();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchUpload = async (file: File) => {
    if (!modelsTrained) {
      setErrorMsg('Train the ML models before scoring batch CSV files.');
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('modelId', selectedModel);
      form.append('threshold', String(threshold));

      const res = await fetch('/api/predict/batch', {
        method: 'POST',
        body: form,
      });
      const text = await res.text();
      let data: Record<string, unknown> = {};
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text.slice(0, 180) || `Batch upload failed (HTTP ${res.status}).`);
      }
      if (!res.ok) throw new Error(String(data.error || 'Batch prediction failed.'));
      setBatchResults(data as typeof batchResults);
      await onPredictionComplete();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Batch upload failed.');
    } finally {
      setLoading(false);
    }
  };

  const downloadSampleBatchTemplate = () => {
    const header = ['transaction_reference', ...FEATURE_KEYS].join(',');
    const legit = legitimatePreset();
    const fraud = suspiciousPreset();
    const row1 = ['SAMPLE-LEGIT-01', ...FEATURE_KEYS.map((k) => legit[k])].join(',');
    const row2 = ['SAMPLE-RISK-02', ...FEATURE_KEYS.map((k) => fraud[k])].join(',');
    const blob = new Blob([[header, row1, row2].join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_batch_transactions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadBatchResultsCsv = () => {
    if (!batchResults) return;
    const header = [
      'transaction_reference',
      'prediction',
      'fraud_probability',
      'risk_level',
      'amount',
      'model_name',
      'created_at',
    ].join(',');
    const lines = batchResults.results.map((r) =>
      [
        r.record.transaction_reference,
        r.record.prediction,
        r.record.fraud_probability,
        r.record.risk_level,
        r.record.amount,
        `"${r.record.model_name}"`,
        r.record.created_at,
      ].join(',')
    );
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'fraud_shield_batch_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Mode & Model Control Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 rounded-xl bg-[#0F172A] border border-slate-800 p-4">
        <div className="inline-flex items-center p-1 rounded-lg bg-slate-900 border border-slate-800 self-start">
          <button
            type="button"
            onClick={() => setMode('single')}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
              mode === 'single'
                ? 'bg-sky-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-slate-100'
            }`}
          >
            Single Transaction
          </button>
          <button
            type="button"
            onClick={() => setMode('batch')}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
              mode === 'batch'
                ? 'bg-sky-500 text-slate-950 font-semibold'
                : 'text-slate-300 hover:text-slate-100'
            }`}
          >
            Batch CSV Scoring
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center gap-2 text-slate-300">
            <span>Model</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelId)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-100 focus:outline-none focus:border-sky-400"
            >
              <option value="tuned_xgboost">Tuned XGBoost</option>
              <option value="decision_tree">Decision Tree</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </label>

          <label className="flex items-center gap-2 text-slate-300">
            <span>Threshold</span>
            <input
              type="number"
              min={0.05}
              max={0.95}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-20 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 font-mono tabular-nums text-slate-100 focus:outline-none focus:border-sky-400"
            />
          </label>
        </div>
      </div>

      {errorMsg && (
        <div className="rounded-xl bg-rose-950/40 border border-rose-800/70 px-4 py-3 text-xs text-rose-200 flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {mode === 'single' ? (
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* Feature Form */}
          <form
            onSubmit={handlePredictSingle}
            className="xl:col-span-7 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-5"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">Transaction Feature Vector</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Enter Time, Amount, and PCA features V1–V28. No card numbers or PII are collected.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setFeatures(legitimatePreset())}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-emerald-300 transition-colors"
                >
                  Load Normal Pattern
                </button>
                <button
                  type="button"
                  onClick={() => setFeatures(suspiciousPreset())}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-rose-300 transition-colors"
                >
                  Load Risk Pattern
                </button>
                <button
                  type="button"
                  onClick={() => setFeatures(defaultFeatureValues())}
                  className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 text-xs text-slate-400 transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <label className="block text-xs">
                <span className="text-slate-400">Reference ID (Optional)</span>
                <input
                  type="text"
                  placeholder="Auto-generated if empty"
                  value={reference}
                  onChange={(e) => setReference(e.target.value)}
                  className="mt-1 w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-slate-100 font-mono text-xs focus:outline-none focus:border-sky-400"
                />
              </label>
              <label className="block text-xs">
                <span className="text-slate-400">Time (Seconds elapsed)</span>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={features.Time}
                  onChange={(e) => setFeatures({ ...features, Time: e.target.value })}
                  className="mt-1 w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-slate-100 font-mono tabular-nums text-xs focus:outline-none focus:border-sky-400"
                />
              </label>
              <label className="block text-xs">
                <span className="text-slate-400">Amount ($)</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={features.Amount}
                  onChange={(e) => setFeatures({ ...features, Amount: e.target.value })}
                  className="mt-1 w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-slate-100 font-mono tabular-nums text-xs focus:outline-none focus:border-sky-400"
                />
              </label>
            </div>

            <div>
              <div className="text-xs text-slate-400 mb-2">PCA Components (V1 – V28)</div>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5">
                {FEATURE_KEYS.slice(1, 29).map((k) => (
                  <label key={k} className="block text-[11px]">
                    <span className="text-slate-400 font-mono">{k}</span>
                    <input
                      type="number"
                      step="any"
                      value={features[k]}
                      onChange={(e) => setFeatures({ ...features, [k]: e.target.value })}
                      className="mt-0.5 w-full rounded bg-slate-900 border border-slate-800 px-2 py-1.5 text-slate-100 font-mono tabular-nums text-xs focus:outline-none focus:border-sky-400"
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-800/80">
              <span className="text-xs text-slate-400">
                {validationError ? (
                  <span className="text-rose-400">{validationError}</span>
                ) : (
                  'All 30 feature inputs validated'
                )}
              </span>
              <button
                type="submit"
                disabled={loading || !!validationError || !modelsTrained}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
              >
                {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                Predict Transaction
              </button>
            </div>
          </form>

          {/* Prediction Result Panel */}
          <div className="xl:col-span-5 rounded-xl bg-[#0F172A] border border-slate-800 p-5 flex flex-col justify-between space-y-5">
            {!singleResult ? (
              <div className="py-16 text-center space-y-2">
                <p className="text-sm font-medium text-slate-300">Ready for Inference</p>
                <p className="text-xs text-slate-400 max-w-sm mx-auto">
                  Submit transaction features on the left to compute fraud probability, risk level,
                  and top contributing deviations.
                </p>
              </div>
            ) : (
              <div className="space-y-5">
                <div
                  className={`rounded-xl p-5 border ${
                    singleResult.record.prediction === 'FRAUDULENT'
                      ? 'bg-rose-950/35 border-rose-700/60'
                      : 'bg-emerald-950/30 border-emerald-700/50'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    {singleResult.record.prediction === 'FRAUDULENT' ? (
                      <ShieldAlert className="w-6 h-6 text-rose-400 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
                    )}
                    <span
                      className={`text-lg font-display font-bold tracking-tight ${
                        singleResult.record.prediction === 'FRAUDULENT'
                          ? 'text-rose-300'
                          : 'text-emerald-300'
                      }`}
                    >
                      {singleResult.record.prediction === 'FRAUDULENT'
                        ? 'FRAUDULENT TRANSACTION'
                        : 'LEGITIMATE TRANSACTION'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 mt-2">{singleResult.confidenceLabel}</p>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3">
                    <div className="text-slate-400">Fraud Probability</div>
                    <div className="text-xl font-mono tabular-nums font-semibold text-slate-100 mt-1">
                      {(singleResult.record.fraud_probability * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3">
                    <div className="text-slate-400">Risk Category</div>
                    <div className="text-xl font-semibold text-slate-100 mt-1">
                      {singleResult.record.risk_level}
                    </div>
                  </div>
                  <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3">
                    <div className="text-slate-400">Selected Model</div>
                    <div className="font-medium text-slate-200 mt-1">
                      {singleResult.record.model_name}
                    </div>
                  </div>
                  <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3">
                    <div className="text-slate-400">Reference & Time</div>
                    <div className="font-mono text-slate-200 mt-1 truncate">
                      {singleResult.record.transaction_reference}
                    </div>
                    <div className="font-mono text-[11px] text-slate-400 mt-0.5">
                      {new Date(singleResult.record.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-xs font-medium text-slate-300 mb-2">
                    Highest Standardized Feature Deviations (z-score)
                  </div>
                  <div className="divide-y divide-slate-800/80 text-xs">
                    {singleResult.topDrivers.map((d) => (
                      <div key={d.feature} className="py-1.5 flex items-center justify-between font-mono tabular-nums">
                        <span className="text-slate-200">{d.feature}</span>
                        <span className="text-slate-400">Raw: {d.raw}</span>
                        <span className={Math.abs(d.zScore) >= 2 ? 'text-amber-300' : 'text-slate-300'}>
                          z = {d.zScore > 0 ? `+${d.zScore}` : d.zScore}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <p className="text-[11px] text-slate-500 border-t border-slate-800/80 pt-3">
              Decision threshold: {threshold.toFixed(2)}. Automated risk scoring supports analyst
              review and does not guarantee fraud prevention.
            </p>
          </div>
        </div>
      ) : (
        /* Batch CSV Mode */
        <div className="space-y-6">
          <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-slate-100">Batch CSV Fraud Screening</h3>
              <p className="text-xs text-slate-400">
                Upload a scoring CSV containing <span className="font-mono text-slate-300">Time, V1–V28, Amount</span> (max 5,000 rows per batch; for training on full <span className="font-mono text-slate-300">creditcard.csv</span>, upload on Dashboard or Settings).
              </p>
              {!modelsTrained && (
                <p className="text-xs text-amber-300 pt-1">
                  Models are not trained yet — upload <span className="font-mono">creditcard.csv</span> or train the ULB Benchmark on Dashboard before batch scoring.
                </p>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={downloadSampleBatchTemplate}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 transition-colors whitespace-nowrap"
              >
                <FileSpreadsheet className="w-4 h-4 text-sky-400" />
                Download Sample CSV
              </button>
              <input
                ref={batchFileRef}
                type="file"
                accept=".csv"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleBatchUpload(f);
                  e.target.value = '';
                }}
                className="hidden"
              />
              <button
                type="button"
                disabled={loading || !modelsTrained}
                onClick={() => batchFileRef.current?.click()}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                Upload Batch CSV
              </button>
            </div>
          </div>

          {batchResults && (
            <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex flex-wrap items-center gap-6 text-xs font-mono tabular-nums">
                  <span className="text-slate-200">Scored: {batchResults.summary.total}</span>
                  <span className="text-rose-400">Flagged Fraudulent: {batchResults.summary.fraudulent}</span>
                  <span className="text-emerald-400">Legitimate: {batchResults.summary.legitimate}</span>
                  <span className="text-slate-400">Deduplicated: {batchResults.summary.deduplicated}</span>
                </div>
                <button
                  type="button"
                  onClick={downloadBatchResultsCsv}
                  className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition-colors whitespace-nowrap"
                >
                  <Download className="w-3.5 h-3.5 text-sky-400" />
                  Download Results CSV
                </button>
              </div>

              <div className="overflow-x-auto max-h-96 overflow-y-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-800 text-slate-400 sticky top-0 bg-[#0F172A]">
                    <tr>
                      <th className="py-2.5 pr-4 font-medium">Reference</th>
                      <th className="py-2.5 pr-4 font-medium">Prediction</th>
                      <th className="py-2.5 pr-4 font-medium text-right">Fraud Prob</th>
                      <th className="py-2.5 pr-4 font-medium">Risk Level</th>
                      <th className="py-2.5 pr-4 font-medium text-right">Amount</th>
                      <th className="py-2.5 font-medium">Confidence Note</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {batchResults.results.map((item) => {
                      const isFraud = item.record.prediction === 'FRAUDULENT';
                      return (
                        <tr
                          key={item.record.transaction_reference}
                          className={isFraud ? 'bg-rose-950/20' : 'hover:bg-slate-900/40'}
                        >
                          <td className="py-2.5 pr-4 font-mono text-slate-200">
                            {item.record.transaction_reference}
                          </td>
                          <td className="py-2.5 pr-4 font-medium">
                            <span className={isFraud ? 'text-rose-400' : 'text-emerald-400'}>
                              {isFraud ? 'FRAUDULENT TRANSACTION' : 'LEGITIMATE TRANSACTION'}
                            </span>
                          </td>
                          <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-200">
                            {(item.record.fraud_probability * 100).toFixed(2)}%
                          </td>
                          <td className="py-2.5 pr-4 text-slate-300">{item.record.risk_level}</td>
                          <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-300">
                            ${item.record.amount.toFixed(2)}
                          </td>
                          <td className="py-2.5 text-slate-400">{item.confidenceLabel}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
