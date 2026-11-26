import React, { useRef, useState } from 'react';
import { Database, Loader2, Upload } from 'lucide-react';

export function DatasetSetupBanner({
  isTraining,
  kaggleFilePresent,
  onUploadCsv,
  onTrainBenchmark,
  onTrainExistingKaggle,
}: {
  isTraining: boolean;
  kaggleFilePresent: boolean;
  onUploadCsv: (file: File) => Promise<void>;
  onTrainBenchmark: () => Promise<void>;
  onTrainExistingKaggle: () => Promise<void>;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [selectedName, setSelectedName] = useState<string>('');

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setSelectedName(f.name);
    try {
      await onUploadCsv(f);
    } finally {
      setSelectedName('');
      e.target.value = '';
    }
  };

  return (
    <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-6 space-y-5">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
        <div className="space-y-2 max-w-2xl">
          <h2 className="text-lg font-semibold text-slate-100">
            Dataset & ML Pipeline Initialization
          </h2>
          <p className="text-sm text-slate-400 leading-relaxed">
            No fake metrics are generated. Upload the Kaggle Credit Card Fraud Detection dataset (
            <span className="font-mono text-slate-200">creditcard.csv</span> — 284,807 transactions,
            31 columns) or train on a local ULB-schema benchmark dataset to fit Logistic Regression,
            Decision Tree, Tuned XGBoost, PCA, and DBSCAN on your data.
          </p>
          <p className="text-xs text-slate-400">
            Kaggle source:{' '}
            <a
              href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
              target="_blank"
              rel="noreferrer"
              className="text-sky-400 hover:text-sky-300 underline underline-offset-2"
            >
              kaggle.com/datasets/mlg-ulb/creditcardfraud
            </a>
            {' '}· Place in <span className="font-mono text-slate-300">data/creditcard.csv</span> or upload below.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            disabled={isTraining}
            onClick={() => fileRef.current?.click()}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
          >
            {isTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            {selectedName ? `Uploading ${selectedName}...` : 'Upload creditcard.csv'}
          </button>

          {kaggleFilePresent && (
            <button
              type="button"
              disabled={isTraining}
              onClick={onTrainExistingKaggle}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 disabled:opacity-50 text-emerald-300 text-xs font-semibold transition-colors whitespace-nowrap"
            >
              Train Local creditcard.csv
            </button>
          )}

          <button
            type="button"
            disabled={isTraining}
            onClick={onTrainBenchmark}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 disabled:opacity-50 text-slate-200 text-xs font-medium transition-colors whitespace-nowrap"
          >
            {isTraining ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Database className="w-4 h-4 text-sky-400" />
            )}
            Train ULB Benchmark (6,000 rows)
          </button>
        </div>
      </div>
    </div>
  );
}
