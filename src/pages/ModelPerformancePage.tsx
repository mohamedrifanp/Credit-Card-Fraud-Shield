import React, { useState } from 'react';
import { Award, Download, Loader2, RefreshCw } from 'lucide-react';
import { ConfusionMatrixGrid, MultiModelCurveChart } from '../components/Charts';
import { DatasetSetupBanner } from '../components/DatasetSetupBanner';
import { AppStatusResponse, ModelId } from '../types';

export function ModelPerformancePage({
  status,
  onUploadCsv,
  onTrainBenchmark,
  onTrainExistingKaggle,
}: {
  status: AppStatusResponse;
  onUploadCsv: (file: File) => Promise<void>;
  onTrainBenchmark: () => Promise<void>;
  onTrainExistingKaggle: () => Promise<void>;
}) {
  const { evaluationReport, isTraining, kaggleFilePresent, datasetSummary } = status;
  const [inspectModel, setInspectModel] = useState<ModelId>('tuned_xgboost');

  if (!evaluationReport.trained || !evaluationReport.models) {
    return (
      <DatasetSetupBanner
        isTraining={isTraining}
        kaggleFilePresent={kaggleFilePresent}
        onUploadCsv={onUploadCsv}
        onTrainBenchmark={onTrainBenchmark}
        onTrainExistingKaggle={onTrainExistingKaggle}
      />
    );
  }

  const models = evaluationReport.models;
  const modelList = [
    models.tuned_xgboost,
    models.decision_tree,
    models.logistic_regression,
  ];
  const activeInspect = models[inspectModel] || models.tuned_xgboost;
  const bestModel = evaluationReport.bestModelId ? models[evaluationReport.bestModelId] : null;

  return (
    <div className="space-y-6">
      {/* Best Model Banner & Report Actions */}
      <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-sky-400 text-xs font-medium">
            <Award className="w-4 h-4" />
            <span>Top Performing Model (Held-Out Stratified Test Set)</span>
          </div>
          <h2 className="text-xl font-semibold text-slate-100">
            {bestModel ? bestModel.modelName : 'Tuned XGBoost'}
            {bestModel && (
              <span className="ml-3 text-sm font-mono tabular-nums font-normal text-slate-300">
                PR-AUC {(bestModel.prAuc * 100).toFixed(2)}% · F1 {(bestModel.f1Score * 100).toFixed(2)}% · Recall{' '}
                {(bestModel.recall * 100).toFixed(2)}%
              </span>
            )}
          </h2>
          <p className="text-xs text-slate-400">{evaluationReport.selectionCriterion}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <a
            href="/api/reports/comparison"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 transition-colors whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5 text-sky-400" />
            Export Comparison CSV
          </a>
          <button
            type="button"
            disabled={isTraining}
            onClick={datasetSummary.source === 'kaggle_csv' ? onTrainExistingKaggle : onTrainBenchmark}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
          >
            {isTraining ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
            Retrain Pipeline
          </button>
        </div>
      </div>

      {/* Model Comparison Matrix */}
      <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Model Benchmark Comparison</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Prioritizes Recall, Precision, F1-Score, and PR-AUC due to severe class imbalance rather
              than Accuracy alone.
            </p>
          </div>
          <span className="text-xs font-mono tabular-nums text-slate-400">
            Trained {evaluationReport.trainedAt ? new Date(evaluationReport.trainedAt).toLocaleString() : ''}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2.5 pr-4 font-medium">Algorithm</th>
                <th className="py-2.5 pr-4 font-medium text-right">Threshold</th>
                <th className="py-2.5 pr-4 font-medium text-right">PR-AUC</th>
                <th className="py-2.5 pr-4 font-medium text-right">ROC-AUC</th>
                <th className="py-2.5 pr-4 font-medium text-right">F1-Score</th>
                <th className="py-2.5 pr-4 font-medium text-right">Recall</th>
                <th className="py-2.5 pr-4 font-medium text-right">Precision</th>
                <th className="py-2.5 pr-4 font-medium text-right">Accuracy</th>
                <th className="py-2.5 font-medium text-right">Train Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {modelList.map((m) => {
                const isBest = evaluationReport.bestModelId === m.modelId;
                return (
                  <tr
                    key={m.modelId}
                    onClick={() => setInspectModel(m.modelId)}
                    className={`cursor-pointer transition-colors ${
                      inspectModel === m.modelId ? 'bg-sky-500/10' : 'hover:bg-slate-900/50'
                    }`}
                  >
                    <td className="py-3 pr-4 font-medium text-slate-100">
                      {m.modelName}
                      {isBest && <span className="ml-2 text-[11px] text-sky-400">· Best</span>}
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums text-slate-300">
                      {m.thresholdUsed.toFixed(2)}
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums font-semibold text-sky-400">
                      {(m.prAuc * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums text-slate-200">
                      {(m.rocAuc * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums text-emerald-400">
                      {(m.f1Score * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums text-slate-200">
                      {(m.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums text-slate-200">
                      {(m.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 pr-4 text-right font-mono tabular-nums text-slate-400">
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 text-right font-mono tabular-nums text-slate-400">
                      {m.trainingTimeMs.toFixed(0)} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* PR & ROC Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <MultiModelCurveChart
            title="Precision-Recall Curves"
            xLabel="Recall"
            yLabel="Precision"
            models={models}
            curveKey="prCurve"
          />
        </div>
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <MultiModelCurveChart
            title="ROC Curves (TPR vs FPR)"
            xLabel="False Positive Rate"
            yLabel="True Positive Rate"
            models={models}
            curveKey="rocCurve"
            showDiagonal
          />
        </div>
      </div>

      {/* Detailed Confusion Matrix & Classification Report */}
      <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-slate-100">
            Confusion Matrix & Classification Report
          </h3>
          <div className="inline-flex items-center p-1 rounded-lg bg-slate-900 border border-slate-800">
            {modelList.map((m) => (
              <button
                key={m.modelId}
                type="button"
                onClick={() => setInspectModel(m.modelId)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  inspectModel === m.modelId
                    ? 'bg-sky-500 text-slate-950 font-semibold'
                    : 'text-slate-300 hover:text-slate-100'
                }`}
              >
                {m.modelName}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-6">
            <ConfusionMatrixGrid
              cm={activeInspect.confusionMatrix}
              modelName={activeInspect.modelName}
            />
          </div>

          <div className="lg:col-span-6 overflow-x-auto">
            <div className="text-xs text-slate-400 mb-3">
              Per-Class Classification Report (Threshold {activeInspect.thresholdUsed.toFixed(2)})
            </div>
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 pr-3 font-medium">Class</th>
                  <th className="py-2 pr-3 font-medium text-right">Precision</th>
                  <th className="py-2 pr-3 font-medium text-right">Recall</th>
                  <th className="py-2 pr-3 font-medium text-right">F1-Score</th>
                  <th className="py-2 font-medium text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-emerald-400 font-medium">
                    Legitimate (0)
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.legitimate.precision * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.legitimate.recall * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.legitimate.f1 * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 text-right text-slate-400">
                    {activeInspect.classificationReport.legitimate.support.toLocaleString()}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-rose-400 font-medium">
                    Fraudulent (1)
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.fraudulent.precision * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.fraudulent.recall * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.fraudulent.f1 * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 text-right text-slate-400">
                    {activeInspect.classificationReport.fraudulent.support.toLocaleString()}
                  </td>
                </tr>
                <tr className="bg-slate-900/40">
                  <td className="py-2.5 pr-3 font-sans text-slate-300 font-medium">Macro Avg</td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.macroAvg.precision * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.macroAvg.recall * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right text-slate-200">
                    {(activeInspect.classificationReport.macroAvg.f1 * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 text-right text-slate-400">
                    {activeInspect.classificationReport.macroAvg.support.toLocaleString()}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Phase 3: Class Imbalance Strategy Comparison */}
      {evaluationReport.imbalanceComparison && evaluationReport.imbalanceComparison.length > 0 && (
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Class Imbalance Handling Comparison (Validation Split)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Compares Original Imbalanced, Class Weights, and SMOTE applied strictly to training
              folds only and scored on the untouched validation split.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 pr-4 font-medium">Strategy</th>
                  <th className="py-2 pr-4 font-medium">Model</th>
                  <th className="py-2 pr-4 font-medium text-right">Train Rows (Fraud)</th>
                  <th className="py-2 pr-4 font-medium text-right">Val PR-AUC</th>
                  <th className="py-2 pr-4 font-medium text-right">Val ROC-AUC</th>
                  <th className="py-2 pr-4 font-medium text-right">Val F1</th>
                  <th className="py-2 pr-4 font-medium text-right">Val Recall</th>
                  <th className="py-2 font-medium text-right">Val Precision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                {evaluationReport.imbalanceComparison.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40">
                    <td className="py-2.5 pr-4 font-sans text-slate-200 font-medium">
                      {row.strategyLabel}
                    </td>
                    <td className="py-2.5 pr-4 font-sans text-slate-300">{row.modelName}</td>
                    <td className="py-2.5 pr-4 text-right text-slate-400">
                      {row.trainRows.toLocaleString()} ({row.trainFraudCount.toLocaleString()})
                    </td>
                    <td className="py-2.5 pr-4 text-right text-sky-400 font-semibold">
                      {(row.valPrAuc * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pr-4 text-right text-slate-200">
                      {(row.valRocAuc * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pr-4 text-right text-emerald-400">
                      {(row.valF1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pr-4 text-right text-slate-200">
                      {(row.valRecall * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 text-right text-slate-200">
                      {(row.valPrecision * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Phase 6: XGBoost Cross-Validation Hyperparameter Tuning */}
      {evaluationReport.cvTrials && evaluationReport.cvTrials.length > 0 && (
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                XGBoost Stratified 3-Fold CV Hyperparameter Tuning
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Hyperparameters and decision threshold selected strictly via cross-validation and
                validation early stopping without touching the test set.
              </p>
            </div>
            {evaluationReport.xgbEarlyStoppedRound !== null && (
              <span className="text-xs font-mono tabular-nums text-sky-400">
                Early stopped at round {evaluationReport.xgbEarlyStoppedRound}
              </span>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono tabular-nums">
              <thead className="border-b border-slate-800 text-slate-400 font-sans">
                <tr>
                  <th className="py-2 pr-3 font-medium">n_estimators</th>
                  <th className="py-2 pr-3 font-medium">max_depth</th>
                  <th className="py-2 pr-3 font-medium">learning_rate</th>
                  <th className="py-2 pr-3 font-medium">subsample</th>
                  <th className="py-2 pr-3 font-medium">colsample_bytree</th>
                  <th className="py-2 pr-3 font-medium">min_child_weight</th>
                  <th className="py-2 pr-3 font-medium text-right">CV PR-AUC</th>
                  <th className="py-2 font-medium text-right">CV F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {evaluationReport.cvTrials.map((trial, i) => {
                  const isChosen =
                    evaluationReport.xgbBestParams &&
                    evaluationReport.xgbBestParams.nEstimators === trial.nEstimators &&
                    evaluationReport.xgbBestParams.maxDepth === trial.maxDepth &&
                    evaluationReport.xgbBestParams.learningRate === trial.learningRate;
                  return (
                    <tr key={i} className={isChosen ? 'bg-sky-500/10' : 'hover:bg-slate-900/40'}>
                      <td className="py-2 pr-3 text-slate-200">
                        {trial.nEstimators}
                        {isChosen && <span className="ml-2 font-sans text-sky-400">· Selected</span>}
                      </td>
                      <td className="py-2 pr-3 text-slate-300">{trial.maxDepth}</td>
                      <td className="py-2 pr-3 text-slate-300">{trial.learningRate}</td>
                      <td className="py-2 pr-3 text-slate-300">{trial.subsample}</td>
                      <td className="py-2 pr-3 text-slate-300">{trial.colsampleBytree}</td>
                      <td className="py-2 pr-3 text-slate-300">{trial.minChildWeight}</td>
                      <td className="py-2 pr-3 text-right text-sky-400">
                        {(trial.cvPrAuc * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 text-right text-emerald-400">
                        {(trial.cvF1 * 100).toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
