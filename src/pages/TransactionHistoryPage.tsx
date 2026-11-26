import React, { useCallback, useEffect, useState } from 'react';
import {
  ArrowDownUp,
  CheckCircle2,
  Download,
  Search,
  ShieldAlert,
  Trash2,
} from 'lucide-react';
import { PredictionRecord } from '../types';

export function TransactionHistoryPage({
  onHistoryMutated,
}: {
  onHistoryMutated: () => Promise<void>;
}) {
  const [records, setRecords] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'LEGITIMATE' | 'FRAUDULENT'>('ALL');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [sortBy, setSortBy] = useState<'created_at' | 'amount' | 'fraud_probability'>('created_at');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [selected, setSelected] = useState<PredictionRecord | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const qs = new URLSearchParams();
      if (search.trim()) qs.set('search', search.trim());
      if (statusFilter !== 'ALL') qs.set('status', statusFilter);
      if (dateFrom) qs.set('dateFrom', dateFrom);
      if (dateTo) qs.set('dateTo', dateTo);
      qs.set('sortBy', sortBy);
      qs.set('sortDir', sortDir);

      const res = await fetch(`/api/history?${qs.toString()}`);
      const data = await res.json();
      setRecords(data.records || []);
    } finally {
      setLoading(false);
    }
  }, [search, statusFilter, dateFrom, dateTo, sortBy, sortDir]);

  useEffect(() => {
    const t = setTimeout(fetchHistory, 180);
    return () => clearTimeout(t);
  }, [fetchHistory]);

  const handleSort = (col: 'created_at' | 'amount' | 'fraud_probability') => {
    if (sortBy === col) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(col);
      setSortDir('desc');
    }
  };

  const handleClear = async () => {
    await fetch('/api/history', { method: 'DELETE' });
    setConfirmClear(false);
    setSelected(null);
    await fetchHistory();
    await onHistoryMutated();
  };

  const parsedSummary = (rec: PredictionRecord) => {
    try {
      return JSON.parse(rec.features_summary) as {
        topDrivers?: Array<{ feature: string; raw: number; zScore: number }>;
      };
    } catch {
      return { topDrivers: [] };
    }
  };

  return (
    <div className="space-y-6">
      {/* Filter & Export Controls */}
      <div className="rounded-xl bg-[#0F172A] border border-slate-800 p-4 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search reference, model, or risk level..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg bg-slate-900 border border-slate-700 pl-9 pr-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-sky-400"
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex items-center p-1 rounded-lg bg-slate-900 border border-slate-800">
              {(['ALL', 'LEGITIMATE', 'FRAUDULENT'] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setStatusFilter(s)}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                    statusFilter === s
                      ? 'bg-sky-500 text-slate-950 font-semibold'
                      : 'text-slate-300 hover:text-slate-100'
                  }`}
                >
                  {s === 'ALL' ? 'All' : s === 'LEGITIMATE' ? 'Legitimate' : 'Fraudulent'}
                </button>
              ))}
            </div>

            <a
              href={`/api/history/export?status=${statusFilter}`}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 transition-colors whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-sky-400" />
              Export CSV
            </a>

            {!confirmClear ? (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-400 hover:text-rose-300 transition-colors whitespace-nowrap"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear
              </button>
            ) : (
              <div className="inline-flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-3 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs text-white font-medium whitespace-nowrap"
                >
                  Confirm Clear
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="px-2.5 py-2 rounded-lg bg-slate-800 text-xs text-slate-300 whitespace-nowrap"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
          <label className="flex items-center gap-2">
            <span>From</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="rounded bg-slate-900 border border-slate-700 px-2.5 py-1 text-slate-200 font-mono"
            />
          </label>
          <label className="flex items-center gap-2">
            <span>To</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="rounded bg-slate-900 border border-slate-700 px-2.5 py-1 text-slate-200 font-mono"
            />
          </label>
          <span className="ml-auto font-mono tabular-nums">
            Showing {records.length.toLocaleString()} SQLite records · No card numbers or PII stored
          </span>
        </div>
      </div>

      {/* Table + Detail Split */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div
          className={`${
            selected ? 'xl:col-span-8' : 'xl:col-span-12'
          } rounded-xl bg-[#0F172A] border border-slate-800 p-5`}
        >
          {loading ? (
            <div className="space-y-2 py-6">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-9 rounded bg-slate-800/50 animate-pulse" />
              ))}
            </div>
          ) : records.length === 0 ? (
            <div className="py-12 text-center text-xs text-slate-500">
              No prediction records match your current filters.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="py-2.5 pr-4 font-medium">Reference</th>
                    <th className="py-2.5 pr-4 font-medium">Status</th>
                    <th className="py-2.5 pr-4 font-medium text-right">
                      <button
                        type="button"
                        onClick={() => handleSort('fraud_probability')}
                        className="inline-flex items-center gap-1 hover:text-slate-200"
                      >
                        Probability <ArrowDownUp className="w-3 h-3" />
                      </button>
                    </th>
                    <th className="py-2.5 pr-4 font-medium text-right">
                      <button
                        type="button"
                        onClick={() => handleSort('amount')}
                        className="inline-flex items-center gap-1 hover:text-slate-200"
                      >
                        Amount <ArrowDownUp className="w-3 h-3" />
                      </button>
                    </th>
                    <th className="py-2.5 pr-4 font-medium">Risk</th>
                    <th className="py-2.5 pr-4 font-medium">Model</th>
                    <th className="py-2.5 font-medium text-right">
                      <button
                        type="button"
                        onClick={() => handleSort('created_at')}
                        className="inline-flex items-center gap-1 hover:text-slate-200"
                      >
                        Created <ArrowDownUp className="w-3 h-3" />
                      </button>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {records.map((rec) => {
                    const isFraud = rec.prediction === 'FRAUDULENT';
                    const isSelected = selected?.id === rec.id;
                    return (
                      <tr
                        key={rec.id}
                        onClick={() => setSelected(rec)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-sky-500/10' : 'hover:bg-slate-900/60'
                        }`}
                      >
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
                            {rec.prediction}
                          </span>
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-200">
                          {(rec.fraud_probability * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-200">
                          ${rec.amount.toFixed(2)}
                        </td>
                        <td className="py-2.5 pr-4 text-slate-300">{rec.risk_level}</td>
                        <td className="py-2.5 pr-4 text-slate-300">{rec.model_name}</td>
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

        {selected && (
          <div className="xl:col-span-4 rounded-xl bg-[#0F172A] border border-slate-800 p-5 space-y-4 self-start">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-100">Transaction Inspection</h3>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="text-xs text-slate-400 hover:text-slate-200"
              >
                Close
              </button>
            </div>

            <div className="text-xs space-y-2 border-b border-slate-800 pb-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Reference</span>
                <span className="font-mono text-slate-200">{selected.transaction_reference}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Outcome</span>
                <span
                  className={`font-semibold ${
                    selected.prediction === 'FRAUDULENT' ? 'text-rose-400' : 'text-emerald-400'
                  }`}
                >
                  {selected.prediction}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Fraud Probability</span>
                <span className="font-mono tabular-nums text-slate-200">
                  {(selected.fraud_probability * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Amount</span>
                <span className="font-mono tabular-nums text-slate-200">
                  ${selected.amount.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Tx Time Offset</span>
                <span className="font-mono tabular-nums text-slate-200">{selected.tx_time}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Model Used</span>
                <span className="text-slate-200">{selected.model_name}</span>
              </div>
            </div>

            <div>
              <div className="text-xs font-medium text-slate-300 mb-2">Top Feature Deviations</div>
              <div className="divide-y divide-slate-800/80 text-xs font-mono tabular-nums">
                {(parsedSummary(selected).topDrivers || []).map((d) => (
                  <div key={d.feature} className="py-1.5 flex justify-between">
                    <span className="text-slate-200">{d.feature}</span>
                    <span className="text-slate-400">Raw {d.raw}</span>
                    <span className={Math.abs(d.zScore) >= 2 ? 'text-amber-300' : 'text-slate-300'}>
                      z {d.zScore > 0 ? `+${d.zScore}` : d.zScore}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
