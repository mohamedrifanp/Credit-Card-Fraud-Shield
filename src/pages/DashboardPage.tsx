import React from 'react';
import { ArrowRight, CheckCircle2, ShieldAlert } from 'lucide-react';
import {
  AmountHistogramChart,
  FraudDistributionChart,
  FraudTrendChart,
} from '../components/Charts';
import { DatasetSetupBanner } from '../components/DatasetSetupBanner';
import { AppStatusResponse, PageId } from '../types';

export function DashboardPage({
  status,
  onNavigate,
  onUploadCsv,
  onTrainBenchmark,
  onTrainExistingKaggle,
}: {
  status: AppStatusResponse;
  onNavigate: (page: PageId) => void;
  onUploadCsv: (file: File) => Promise<void>;
  onTrainBenchmark: () => Promise<void>;
  onTrainExistingKaggle: () => Promise<void>;
}) {
  const { datasetSummary, evaluationReport, settings, recentPredictions, predictionStats } = status;
  const activeEval = evaluationReport.models?.[settings.activeModel] ?? null;

  return (
    <div className="space-y-6">
      {!datasetSummary.loaded && (
        <DatasetSetupBanner
          isTraining={status.isTraining}
          kaggleFilePresent={status.kaggleFilePresent}
          onUploadCsv={onUploadCsv}
          onTrainBenchmark={onTrainBenchmark}
          onTrainExistingKaggle={onTrainExistingKaggle}
        />
      )}

      {/* Primary KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Total Dataset Transactions</div>
          <div className="text-2xl font-semibold font-mono tabular-nums text-slate-100 mt-1.5">
            {datasetSummary.loaded ? datasetSummary.totalRows.toLocaleString() : '—'}
          </div>
          <div className="text-xs text-slate-400 mt-2 truncate">
            {datasetSummary.loaded
              ? `${datasetSummary.filename} · ${datasetSummary.featureCount} features`
              : 'Awaiting CSV dataset upload'}
          </div>
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Legitimate Transactions</div>
          <div className="text-2xl font-semibold font-mono tabular-nums text-emerald-400 mt-1.5">
            {datasetSummary.loaded ? datasetSummary.legitimateCount.toLocaleString() : '—'}
          </div>
          <div className="text-xs text-slate-400 mt-2 font-mono tabular-nums">
            {datasetSummary.loaded
              ? `${(100 - datasetSummary.fraudPercentage).toFixed(2)}% of dataset`
              : 'Class 0 baseline'}
          </div>
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Fraudulent Transactions</div>
          <div className="text-2xl font-semibold font-mono tabular-nums text-rose-400 mt-1.5">
            {datasetSummary.loaded ? datasetSummary.fraudCount.toLocaleString() : '—'}
          </div>
          <div className="text-xs text-slate-400 mt-2 font-mono tabular-nums">
            {datasetSummary.loaded
              ? `${datasetSummary.fraudPercentage.toFixed(4)}% fraud prevalence`
              : 'Class 1 target'}
          </div>
        </div>

        <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5">
          <div className="text-xs text-slate-400">
            {activeEval ? `${activeEval.modelName} Recall` : 'Active Model Recall'}
          </div>
          <div className="text-2xl font-semibold font-mono tabular-nums text-sky-400 mt-1.5">
            {activeEval ? `${(activeEval.recall * 100).toFixed(1)}%` : '—'}
          </div>
          <div className="text-xs text-slate-400 mt-2 font-mono tabular-nums">
            {activeEval
              ? `PR-AUC ${(activeEval.prAuc * 100).toFixed(1)}% · F1 ${(activeEval.f1Score * 100).toFixed(1)}%`
              : 'Train pipeline to evaluate'}
          </div>
        </div>
      </div>

      {/* Distribution & Amount Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Fraud Class Distribution</h3>
            <span className="text-xs text-slate-400 font-mono tabular-nums">
              {datasetSummary.loaded ? `Duplicates: ${datasetSummary.duplicateRows}` : 'Unloaded'}
            </span>
          </div>
          {datasetSummary.loaded ? (
            <FraudDistributionChart
              legitimateCount={datasetSummary.legitimateCount}
              fraudCount={datasetSummary.fraudCount}
            />
          ) : (
            <p className="text-xs text-slate-500 py-10 text-center">
              Load `creditcard.csv` or train the benchmark dataset to inspect class distribution.
            </p>
          )}
        </div>

        <div className="lg:col-span-7 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Transaction Amount Distribution</h3>
            <span className="text-xs text-slate-400 font-mono tabular-nums">
              {datasetSummary.loaded
                ? `Fraud Avg $${datasetSummary.amountStats.fraudMean.toFixed(2)} · Legit Avg $${datasetSummary.amountStats.legitMean.toFixed(2)}`
                : ''}
            </span>
          </div>
          <AmountHistogramChart buckets={datasetSummary.amountHistogram} />
        </div>
      </div>

      {/* Fraud Trends + Model Status */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Temporal Fraud Trends</h3>
            <button
              type="button"
              onClick={() => onNavigate('fraud-analytics')}
              className="inline-flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300"
            >
              Deep Analytics <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
          <FraudTrendChart trends={datasetSummary.timeTrends} />
        </div>

        <div className="lg:col-span-5 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Model & Database Status</h3>
            <button
              type="button"
              onClick={() => onNavigate('model-performance')}
              className="inline-flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300"
            >
              Compare Models <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="divide-y divide-slate-800/80 text-xs">
            {(['tuned_xgboost', 'decision_tree', 'logistic_regression'] as const).map((id) => {
              const m = evaluationReport.models?.[id];
              const isActive = settings.activeModel === id;
              return (
                <div key={id} className="py-3 flex items-center justify-between gap-2">
                  <div>
                    <div className="font-medium text-slate-200">
                      {m ? m.modelName : id.replace('_', ' ')}
                      {isActive && (
                        <span className="ml-2 text-[11px] text-sky-400 font-normal">· Active</span>
                      )}
                    </div>
                    <div className="text-slate-400 font-mono tabular-nums mt-0.5">
                      {m
                        ? `PR-AUC ${(m.prAuc * 100).toFixed(1)}% · F1 ${(m.f1Score * 100).toFixed(1)}% · Recall ${(m.recall * 100).toFixed(1)}%`
                        : 'Not trained yet'}
                    </div>
                  </div>
                  <span
                    className={`font-mono text-[11px] ${
                      status.modelsReady[id] ? 'text-emerald-400' : 'text-slate-500'
                    }`}
                  >
                    {status.modelsReady[id] ? 'Ready' : 'Untrained'}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>Stored SQLite Predictions</span>
            <span className="font-mono tabular-nums text-slate-200">
              {predictionStats.totalPredictions.toLocaleString()} total ·{' '}
              {predictionStats.fraudulentPredictions.toLocaleString()} flagged
            </span>
          </div>
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Recent Predictions</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Latest inference evaluations persisted in SQLite (`database/transactions.db`).
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => onNavigate('fraud-detection')}
              className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
            >
              Score Transaction
            </button>
            <button
              type="button"
              onClick={() => onNavigate('transaction-history')}
              className="text-xs text-slate-300 hover:text-slate-100 whitespace-nowrap"
            >
              View All
            </button>
          </div>
        </div>

        {recentPredictions.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
            No predictions recorded yet. Run single or batch inference in Fraud Detection.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2.5 pr-4 font-medium">Reference</th>
                  <th className="py-2.5 pr-4 font-medium">Prediction</th>
                  <th className="py-2.5 pr-4 font-medium text-right">Fraud Prob</th>
                  <th className="py-2.5 pr-4 font-medium text-right">Amount</th>
                  <th className="py-2.5 pr-4 font-medium">Model</th>
                  <th className="py-2.5 pr-4 font-medium">Risk Level</th>
                  <th className="py-2.5 font-medium text-right">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {recentPredictions.map((rec) => {
                  const isFraud = rec.prediction === 'FRAUDULENT';
                  return (
                    <tr key={rec.id} className="hover:bg-slate-900/50 transition-colors">
                      <td className="py-2.5 pr-4 font-mono text-slate-200">
                        {rec.transaction_reference}
                      </td>
                      <td className="py-2.5 pr-4">
                        <span
                          className={`inline-flex items-center gap-1.5 font-medium ${
                            isFraud ? 'text-rose-400' : 'text-emerald-400'
                          }`}
                        >
                          {isFraud ? (
                            <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                          ) : (
                            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                          )}
                          {isFraud ? 'FRAUDULENT' : 'LEGITIMATE'}
                        </span>
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-200">
                        {(rec.fraud_probability * 100).toFixed(2)}%
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-300">
                        ${rec.amount.toFixed(2)}
                      </td>
                      <td className="py-2.5 pr-4 text-slate-300">{rec.model_name}</td>
                      <td className="py-2.5 pr-4 text-slate-300">{rec.risk_level}</td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-slate-400">
                        {new Date(rec.created_at).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
