import React, { useState } from 'react';
import { Download } from 'lucide-react';
import {
  DbscanScatterChart,
  FeatureImportanceChart,
  FraudTrendChart,
  PcaVarianceChart,
} from '../components/Charts';
import { DatasetSetupBanner } from '../components/DatasetSetupBanner';
import { AppStatusResponse, ModelId } from '../types';

export function FraudAnalyticsPage({
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
  const { datasetSummary, edaReport, evaluationReport, isTraining, kaggleFilePresent } = status;
  const [importanceModel, setImportanceModel] = useState<ModelId>('tuned_xgboost');

  if (!datasetSummary.loaded || !evaluationReport.trained || !evaluationReport.models) {
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

  const pca = evaluationReport.pcaExperiment;
  const dbscan = evaluationReport.dbscanExperiment;
  const impItems = evaluationReport.models[importanceModel]?.featureImportances || [];

  return (
    <div className="space-y-6">
      {/* Phase 1: EDA Summary & Duplicate Investigation */}
      {edaReport?.generated && (
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                Exploratory Data Analysis (Phase 1 Report)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Shape:{' '}
                <span className="font-mono tabular-nums text-slate-200">
                  {edaReport.shape.rows.toLocaleString()} × {edaReport.shape.columns}
                </span>{' '}
                · Missing values:{' '}
                <span className="font-mono tabular-nums text-slate-200">
                  {edaReport.missingValuesTotal}
                </span>{' '}
                · Duplicates investigated:{' '}
                <span className="font-mono tabular-nums text-amber-300">
                  {edaReport.duplicateInvestigation.totalDuplicates}
                </span>{' '}
                ({edaReport.duplicateInvestigation.legitDuplicates} legit,{' '}
                {edaReport.duplicateInvestigation.fraudDuplicates} fraud)
              </p>
            </div>
            <a
              href="/api/reports/eda"
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 transition-colors whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-sky-400" />
              Export EDA JSON
            </a>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            {edaReport.duplicateInvestigation.handlingDecision} Deduplicated dataset rows:{' '}
            <span className="font-mono tabular-nums text-slate-200">
              {edaReport.duplicateInvestigation.rowsAfterDeduplication.toLocaleString()}
            </span>
            .
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5">
              <div className="text-xs font-medium text-slate-300 mb-2">
                Strongest Feature Correlations with Class (Pearson r)
              </div>
              <div className="divide-y divide-slate-800/70 text-xs font-mono tabular-nums">
                {edaReport.topCorrelations.slice(0, 8).map((c) => (
                  <div key={c.feature} className="py-1.5 flex items-center justify-between">
                    <span className="text-slate-200">{c.feature}</span>
                    <span className={c.correlation < 0 ? 'text-rose-400' : 'text-sky-400'}>
                      {c.correlation > 0 ? `+${c.correlation.toFixed(4)}` : c.correlation.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="lg:col-span-7 overflow-x-auto max-h-64 overflow-y-auto">
              <div className="text-xs font-medium text-slate-300 mb-2">
                Column Data Types & Summary Statistics
              </div>
              <table className="w-full text-left text-xs font-mono tabular-nums">
                <thead className="border-b border-slate-800 text-slate-400 font-sans sticky top-0 bg-[#0F172A]">
                  <tr>
                    <th className="py-1.5 pr-3 font-medium">Column</th>
                    <th className="py-1.5 pr-3 font-medium">Dtype</th>
                    <th className="py-1.5 pr-3 font-medium text-right">Mean</th>
                    <th className="py-1.5 pr-3 font-medium text-right">Std</th>
                    <th className="py-1.5 pr-3 font-medium text-right">Min</th>
                    <th className="py-1.5 font-medium text-right">Max</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {edaReport.columns.slice(0, 12).map((col) => (
                    <tr key={col.name}>
                      <td className="py-1.5 pr-3 text-slate-200">{col.name}</td>
                      <td className="py-1.5 pr-3 text-slate-400">{col.dtype}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-300">{col.mean}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-300">{col.std}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-400">{col.min}</td>
                      <td className="py-1.5 text-right text-slate-400">{col.max}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Amount & Prevalence Analytics Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Fraud Prevalence Rate</div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-rose-400 mt-1.5">
            {datasetSummary.fraudPercentage.toFixed(4)}%
          </div>
          <div className="text-xs text-slate-400 mt-1.5 font-mono tabular-nums">
            {datasetSummary.fraudCount.toLocaleString()} / {datasetSummary.totalRows.toLocaleString()} tx
          </div>
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Fraud Mean / Median Amount</div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-slate-100 mt-1.5">
            ${datasetSummary.amountStats.fraudMean.toFixed(2)}
          </div>
          <div className="text-xs text-slate-400 mt-1.5 font-mono tabular-nums">
            Median ${datasetSummary.amountStats.fraudMedian.toFixed(2)} · Max $
            {datasetSummary.amountStats.fraudMax.toFixed(2)}
          </div>
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Legitimate Mean / Median Amount</div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-emerald-400 mt-1.5">
            ${datasetSummary.amountStats.legitMean.toFixed(2)}
          </div>
          <div className="text-xs text-slate-400 mt-1.5 font-mono tabular-nums">
            Median ${datasetSummary.amountStats.legitMedian.toFixed(2)} · Max $
            {datasetSummary.amountStats.legitMax.toFixed(2)}
          </div>
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">DBSCAN Noise Outliers</div>
          <div className="text-2xl font-mono tabular-nums font-semibold text-amber-400 mt-1.5">
            {dbscan ? `${dbscan.noisePercentage.toFixed(2)}%` : '—'}
          </div>
          <div className="text-xs text-slate-400 mt-1.5 font-mono tabular-nums">
            {dbscan
              ? `${dbscan.noiseCount} outliers (${dbscan.noiseFraudCount} fraud overlap)`
              : 'Not computed'}
          </div>
        </div>
      </div>

      {/* Feature Importance + Temporal Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-100">Feature Importance (Top 12)</h3>
            <select
              value={importanceModel}
              onChange={(e) => setImportanceModel(e.target.value as ModelId)}
              className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-sky-400"
            >
              <option value="tuned_xgboost">Tuned XGBoost</option>
              <option value="decision_tree">Decision Tree</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </div>
          <FeatureImportanceChart items={impItems} />
        </div>

        <div className="lg:col-span-7 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-3">
          <h3 className="text-sm font-semibold text-slate-100">Transaction & Fraud Rate Trends</h3>
          <FraudTrendChart trends={datasetSummary.timeTrends} />
          <p className="text-xs text-slate-400 leading-relaxed pt-2">
            Fraud prevalence fluctuates noticeably across the 48-hour ULB observation window, with
            lower-volume off-peak windows often exhibiting higher relative fraud percentages.
          </p>
        </div>
      </div>

      {/* PCA Dimensionality Experiment */}
      {pca && (
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                PCA Dimensionality Reduction Experiment
              </h3>
              <p className="text-xs text-slate-400 mt-0.5 max-w-3xl">{pca.note}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            <div className="lg:col-span-7">
              <PcaVarianceChart
                explainedRatio={pca.explainedVarianceRatio}
                cumulative={pca.cumulativeVariance}
                components95={pca.componentsFor95Pct}
              />
            </div>

            <div className="lg:col-span-5 overflow-x-auto">
              <div className="text-xs text-slate-400 mb-2">
                Logistic Regression: Full Features vs 95%-Variance PCA
              </div>
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="py-2 pr-3 font-medium">Metric</th>
                    <th className="py-2 pr-3 font-medium text-right">
                      Before PCA ({pca.beforePca.dimensions}D)
                    </th>
                    <th className="py-2 font-medium text-right">
                      After PCA ({pca.afterPca.dimensions}D)
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                  <tr>
                    <td className="py-2 pr-3 font-sans text-slate-300">PR-AUC</td>
                    <td className="py-2 pr-3 text-right text-slate-200">
                      {(pca.beforePca.prAuc * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 text-right text-sky-400">
                      {(pca.afterPca.prAuc * 100).toFixed(2)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-3 font-sans text-slate-300">ROC-AUC</td>
                    <td className="py-2 pr-3 text-right text-slate-200">
                      {(pca.beforePca.rocAuc * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 text-right text-sky-400">
                      {(pca.afterPca.rocAuc * 100).toFixed(2)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-3 font-sans text-slate-300">F1-Score</td>
                    <td className="py-2 pr-3 text-right text-slate-200">
                      {(pca.beforePca.f1Score * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 text-right text-sky-400">
                      {(pca.afterPca.f1Score * 100).toFixed(2)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-3 font-sans text-slate-300">Recall</td>
                    <td className="py-2 pr-3 text-right text-slate-200">
                      {(pca.beforePca.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 text-right text-sky-400">
                      {(pca.afterPca.recall * 100).toFixed(2)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-3 font-sans text-slate-300">Precision</td>
                    <td className="py-2 pr-3 text-right text-slate-200">
                      {(pca.beforePca.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 text-right text-sky-400">
                      {(pca.afterPca.precision * 100).toFixed(2)}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* DBSCAN Exploratory Anomaly Detection */}
      {dbscan && (
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              DBSCAN Exploratory Anomaly Detection
            </h3>
            <p className="text-xs text-slate-400 mt-0.5 max-w-3xl">{dbscan.note}</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-8">
              <DbscanScatterChart points={dbscan.points} />
            </div>

            <div className="lg:col-span-4 space-y-3 text-xs">
              <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3.5 space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-400">Neighborhood Radius (eps)</span>
                  <span className="font-mono tabular-nums text-slate-200">{dbscan.eps}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Min Samples</span>
                  <span className="font-mono tabular-nums text-slate-200">{dbscan.minSamples}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Sampled Transactions</span>
                  <span className="font-mono tabular-nums text-slate-200">
                    {dbscan.sampledCount.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Dense Clusters Found</span>
                  <span className="font-mono tabular-nums text-slate-200">{dbscan.clusterCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Noise Outliers (-1)</span>
                  <span className="font-mono tabular-nums text-amber-300">
                    {dbscan.noiseCount} ({dbscan.noisePercentage}%)
                  </span>
                </div>
              </div>

              <div className="rounded-lg bg-slate-900/70 border border-slate-800 p-3.5 space-y-2">
                <div className="text-slate-300 font-medium">Post-Hoc Fraud Label Overlap</div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Frauds Captured in Noise</span>
                  <span className="font-mono tabular-nums text-rose-400">
                    {dbscan.noiseFraudCount} / {dbscan.totalFraudInSample} (
                    {(dbscan.noiseFraudRecall * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Noise Precision vs Fraud</span>
                  <span className="font-mono tabular-nums text-slate-200">
                    {(dbscan.noiseFraudPrecision * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* High-Risk Dataset Transactions */}
      <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            High-Magnitude Fraudulent Samples in Dataset
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Actual high-deviation Class 1 instances sorted by |V14| + |V17| magnitude.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2 pr-4 font-medium">Row</th>
                <th className="py-2 pr-4 font-medium text-right">Time (s)</th>
                <th className="py-2 pr-4 font-medium text-right">Amount</th>
                <th className="py-2 pr-4 font-medium text-right">V14</th>
                <th className="py-2 pr-4 font-medium text-right">V17</th>
                <th className="py-2 pr-4 font-medium text-right">V12</th>
                <th className="py-2 font-medium text-right">Label</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
              {datasetSummary.highRiskSamples.map((s) => (
                <tr key={s.rowIndex} className="hover:bg-slate-900/40">
                  <td className="py-2 pr-4 text-slate-300">#{s.rowIndex}</td>
                  <td className="py-2 pr-4 text-right text-slate-300">{s.time}</td>
                  <td className="py-2 pr-4 text-right text-slate-200">${s.amount.toFixed(2)}</td>
                  <td className="py-2 pr-4 text-right text-rose-400">{s.v14}</td>
                  <td className="py-2 pr-4 text-right text-rose-400">{s.v17}</td>
                  <td className="py-2 pr-4 text-right text-amber-300">{s.v12}</td>
                  <td className="py-2 text-right text-rose-400 font-sans font-medium">
                    Fraud (1)
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
