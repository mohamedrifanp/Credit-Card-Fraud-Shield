import React, { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, Save } from 'lucide-react';
import { ThresholdImpactChart } from '../components/Charts';
import { DatasetSetupBanner } from '../components/DatasetSetupBanner';
import { AppStatusResponse, ModelId } from '../types';

export function SettingsPage({
  status,
  onSettingsSaved,
  onUploadCsv,
  onTrainBenchmark,
  onTrainExistingKaggle,
}: {
  status: AppStatusResponse;
  onSettingsSaved: () => Promise<void>;
  onUploadCsv: (file: File) => Promise<void>;
  onTrainBenchmark: () => Promise<void>;
  onTrainExistingKaggle: () => Promise<void>;
}) {
  const [activeModel, setActiveModel] = useState<ModelId>(status.settings.activeModel);
  const [threshold, setThreshold] = useState<number>(status.settings.threshold);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ ok: boolean; msg: string } | null>(null);

  useEffect(() => {
    setActiveModel(status.settings.activeModel);
    setThreshold(status.settings.threshold);
  }, [status.settings.activeModel, status.settings.threshold]);

  const activeEval = status.evaluationReport.models?.[activeModel] || null;

  // Find closest sweep point to preview live precision/recall impact
  const closestSweep = activeEval?.thresholdSweep.reduce((prev, curr) =>
    Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
  );

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!Number.isFinite(threshold) || threshold < 0.05 || threshold > 0.95) {
      setFeedback({ ok: false, msg: 'Threshold must be between 0.05 and 0.95.' });
      return;
    }
    setSaving(true);
    setFeedback(null);
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ activeModel, threshold }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to save settings.');
      setFeedback({ ok: true, msg: 'Active model and threshold saved to SQLite.' });
      await onSettingsSaved();
    } catch (err) {
      setFeedback({
        ok: false,
        msg: err instanceof Error ? err.message : 'Unable to save settings.',
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Model & Threshold Configuration */}
      <form
        onSubmit={handleSave}
        className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-5"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Inference Model & Decision Threshold
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Adjusting the threshold alters the sensitivity-specificity balance for real-time and
              batch predictions.
            </p>
          </div>
          <button
            type="submit"
            disabled={saving}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
          >
            {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
            Save Configuration
          </button>
        </div>

        {feedback && (
          <div
            className={`rounded-lg px-3.5 py-2.5 text-xs flex items-center gap-2 ${
              feedback.ok
                ? 'bg-emerald-950/40 border border-emerald-800/60 text-emerald-300'
                : 'bg-rose-950/40 border border-rose-800/60 text-rose-300'
            }`}
          >
            {feedback.ok && <CheckCircle2 className="w-4 h-4 shrink-0" />}
            <span>{feedback.msg}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-5 space-y-4">
            <label className="block text-xs">
              <span className="text-slate-300 font-medium">Active ML Model</span>
              <select
                value={activeModel}
                onChange={(e) => setActiveModel(e.target.value as ModelId)}
                className="mt-1.5 w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-slate-100 text-xs focus:outline-none focus:border-sky-400"
              >
                <option value="tuned_xgboost">Tuned XGBoost (Gradient Boosted Trees)</option>
                <option value="decision_tree">Decision Tree (Class-Weighted CART)</option>
                <option value="logistic_regression">Logistic Regression (L2 + Class Weighting)</option>
              </select>
            </label>

            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-medium">Fraud Classification Threshold</span>
                <span className="font-mono tabular-nums text-sky-400 font-semibold">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-sky-400 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                <span>0.05 (High Recall)</span>
                <span>0.50 (Balanced)</span>
                <span>0.95 (High Precision)</span>
              </div>
            </div>

            {closestSweep && (
              <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3.5 space-y-2 text-xs">
                <div className="text-slate-400">
                  Projected Impact on Held-Out Test Set (at {closestSweep.threshold.toFixed(2)})
                </div>
                <div className="grid grid-cols-3 gap-2 pt-1 font-mono tabular-nums">
                  <div>
                    <div className="text-[11px] text-slate-500">Precision</div>
                    <div className="text-sm text-sky-400 font-semibold">
                      {(closestSweep.precision * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">Recall</div>
                    <div className="text-sm text-emerald-400 font-semibold">
                      {(closestSweep.recall * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">F1-Score</div>
                    <div className="text-sm text-amber-400 font-semibold">
                      {(closestSweep.f1 * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="lg:col-span-7">
            {activeEval ? (
              <ThresholdImpactChart
                sweep={activeEval.thresholdSweep}
                activeThreshold={threshold}
              />
            ) : (
              <div className="py-12 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
                Train the ML pipeline to inspect threshold sensitivity on test data.
              </div>
            )}
          </div>
        </div>
      </form>

      {/* Dataset Upload & Retraining */}
      <DatasetSetupBanner
        isTraining={status.isTraining}
        kaggleFilePresent={status.kaggleFilePresent}
        onUploadCsv={onUploadCsv}
        onTrainBenchmark={onTrainBenchmark}
        onTrainExistingKaggle={onTrainExistingKaggle}
      />

      {/* System, Preprocessing & SQLite Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-3 text-xs">
          <h3 className="text-sm font-semibold text-slate-100">
            Preprocessing & Data Leakage Safeguards
          </h3>
          {status.preprocessing ? (
            <div className="divide-y divide-slate-800/70 font-mono tabular-nums">
              <div className="py-2 flex justify-between">
                <span className="font-sans text-slate-400">Stratified Split (Train / Val / Test)</span>
                <span className="text-slate-200">
                  {status.preprocessing.trainRows.toLocaleString()} /{' '}
                  {status.preprocessing.valRows.toLocaleString()} /{' '}
                  {status.preprocessing.testRows.toLocaleString()}
                </span>
              </div>
              <div className="py-2 flex justify-between">
                <span className="font-sans text-slate-400">Fraud Cases per Split</span>
                <span className="text-slate-200">
                  {status.preprocessing.trainFraudCount} / {status.preprocessing.valFraudCount} /{' '}
                  {status.preprocessing.testFraudCount}
                </span>
              </div>
              <div className="py-2 flex justify-between">
                <span className="font-sans text-slate-400">Time RobustScaler (Median / IQR)</span>
                <span className="text-slate-200">
                  {status.preprocessing.timeMedian.toFixed(1)} /{' '}
                  {status.preprocessing.timeIqr.toFixed(1)}
                </span>
              </div>
              <div className="py-2 flex justify-between">
                <span className="font-sans text-slate-400">Amount RobustScaler (Median / IQR)</span>
                <span className="text-slate-200">
                  ${status.preprocessing.amountMedian.toFixed(2)} / $
                  {status.preprocessing.amountIqr.toFixed(2)}
                </span>
              </div>
              <div className="py-2 flex justify-between">
                <span className="font-sans text-slate-400">PCA Features V1–V28</span>
                <span className="text-emerald-400 font-sans">
                  {status.preprocessing.vFeaturesPreserved ? 'Preserved Intact' : 'Standardized'}
                </span>
              </div>
              <div className="py-2 flex justify-between">
                <span className="font-sans text-slate-400">Exact Duplicates Removed</span>
                <span className="text-slate-200">
                  {status.preprocessing.duplicatesRemoved.toLocaleString()}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-slate-500 py-4">
              Scalers fit strictly on training folds once a dataset is loaded.
            </p>
          )}
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-3 text-xs">
          <h3 className="text-sm font-semibold text-slate-100">Database & Saved Model Artifacts</h3>
          <div className="divide-y divide-slate-800/70">
            <div className="py-2 flex justify-between gap-4">
              <span className="text-slate-400">Application</span>
              <span className="text-slate-200 font-medium">
                Credit Card Fraud Shield · Intelligent Transaction Security
              </span>
            </div>
            <div className="py-2 flex justify-between gap-4">
              <span className="text-slate-400">SQLite Engine</span>
              <span className="text-slate-200 font-mono truncate">
                {status.databaseStats.dbPath}
              </span>
            </div>
            <div className="py-2 flex justify-between gap-4">
              <span className="text-slate-400">Stored Predictions</span>
              <span className="text-slate-200 font-mono tabular-nums">
                {status.databaseStats.totalRecords.toLocaleString()} records
              </span>
            </div>
            {status.modelMetadata && (
              <div className="py-2 flex justify-between gap-4">
                <span className="text-slate-400">Saved Best Artifact</span>
                <span className="text-sky-400 font-mono truncate">
                  {status.modelMetadata.bestModelFile} (Threshold{' '}
                  {status.modelMetadata.selectedThreshold.toFixed(2)})
                </span>
              </div>
            )}
            <div className="py-2 flex justify-between gap-4">
              <span className="text-slate-400">Privacy Guarantee</span>
              <span className="text-emerald-400">
                No card numbers, CVVs, or cardholder PII stored
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
