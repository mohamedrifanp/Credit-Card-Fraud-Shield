import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  History,
  LayoutDashboard,
  Menu,
  ScanSearch,
  Settings,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import { ShieldLogo } from './components/ShieldLogo';
import { DashboardPage } from './pages/DashboardPage';
import { FraudAnalyticsPage } from './pages/FraudAnalyticsPage';
import { FraudDetectionPage } from './pages/FraudDetectionPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { SettingsPage } from './pages/SettingsPage';
import { TransactionHistoryPage } from './pages/TransactionHistoryPage';
import { AppStatusResponse, PageId } from './types';

const NAV_ITEMS: Array<{
  id: PageId;
  label: string;
  subtitle: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    subtitle: 'Dataset overview, fraud distribution, and recent predictions',
    icon: LayoutDashboard,
  },
  {
    id: 'fraud-detection',
    label: 'Fraud Detection',
    subtitle: 'Interactive single-transaction and batch CSV fraud scoring',
    icon: ScanSearch,
  },
  {
    id: 'transaction-history',
    label: 'Transaction History',
    subtitle: 'Searchable SQLite audit log and CSV export',
    icon: History,
  },
  {
    id: 'model-performance',
    label: 'Model Performance',
    subtitle: 'Logistic Regression, Decision Tree, and Tuned XGBoost benchmarks',
    icon: BarChart3,
  },
  {
    id: 'fraud-analytics',
    label: 'Fraud Analytics',
    subtitle: 'PCA variance experiment, DBSCAN anomaly detection, and trends',
    icon: Activity,
  },
  {
    id: 'settings',
    label: 'Settings',
    subtitle: 'Active model, classification threshold, and dataset management',
    icon: Settings,
  },
];

export default function App() {
  const [page, setPage] = useState<PageId>('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [status, setStatus] = useState<AppStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const showToast = useCallback((type: 'success' | 'error', text: string) => {
    setToast({ type, text });
    setTimeout(() => {
      setToast((curr) => (curr?.text === text ? null : curr));
    }, 5000);
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Unable to load application status.');
      const data = (await res.json()) as AppStatusResponse;
      setStatus(data);
    } catch (err) {
      showToast('error', err instanceof Error ? err.message : 'Backend connection error.');
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const [uploadProgress, setUploadProgress] = useState<string | null>(null);

  const safeReadJson = async (res: Response) => {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return { error: text.slice(0, 180) || `HTTP ${res.status} error` };
    }
  };

  const handleUploadCsv = async (file: File) => {
    setStatus((s) => (s ? { ...s, isTraining: true } : s));
    setUploadProgress(`Preparing ${file.name}...`);
    try {
      const CHUNK_SIZE = 4 * 1024 * 1024; // 4 MB chunks safely stay below Cloud Run's 32 MB limit
      if (file.size <= CHUNK_SIZE) {
        setUploadProgress(`Uploading ${file.name}...`);
        const form = new FormData();
        form.append('file', file);
        const res = await fetch('/api/dataset/upload', { method: 'POST', body: form });
        const data = await safeReadJson(res);
        if (!res.ok) throw new Error(data.error || 'CSV upload and training failed.');
      } else {
        const initRes = await fetch('/api/dataset/upload/init', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name }),
        });
        const initData = await safeReadJson(initRes);
        if (!initRes.ok) throw new Error(initData.error || 'Failed to initialize upload.');

        const uploadId = initData.uploadId as string;
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        for (let idx = 0; idx < totalChunks; idx++) {
          const start = idx * CHUNK_SIZE;
          const end = Math.min(file.size, start + CHUNK_SIZE);
          const pct = Math.round(((idx + 1) / totalChunks) * 100);
          setUploadProgress(`Uploading ${file.name} (${pct}%)...`);

          const buf = await file.slice(start, end).arrayBuffer();
          const chunkRes = await fetch('/api/dataset/upload/chunk', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/octet-stream',
              'X-Upload-Id': uploadId,
            },
            body: buf,
          });
          const chunkData = await safeReadJson(chunkRes);
          if (!chunkRes.ok) throw new Error(chunkData.error || `Failed on chunk ${idx + 1}.`);
        }

        setUploadProgress('Parsing dataset & training ML models...');
        const doneRes = await fetch('/api/dataset/upload/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ uploadId }),
        });
        const doneData = await safeReadJson(doneRes);
        if (!doneRes.ok) throw new Error(doneData.error || 'Dataset processing failed.');
      }

      await fetchStatus();
      showToast('success', 'Dataset uploaded and all ML models trained successfully.');
    } catch (err) {
      await fetchStatus();
      showToast('error', err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setUploadProgress(null);
    }
  };

  const handleTrainMode = async (mode: 'benchmark' | 'kaggle') => {
    setStatus((s) => (s ? { ...s, isTraining: true } : s));
    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Training failed.');
      await fetchStatus();
      showToast(
        'success',
        mode === 'benchmark'
          ? 'Trained all models on 6,000-row ULB benchmark dataset.'
          : 'Trained all models on local creditcard.csv.'
      );
    } catch (err) {
      await fetchStatus();
      showToast('error', err instanceof Error ? err.message : 'Training failed.');
    }
  };

  const activeMeta = NAV_ITEMS.find((n) => n.id === page) || NAV_ITEMS[0];

  const handleNav = (id: PageId) => {
    setPage(id);
    setMobileMenuOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#080D1A] text-slate-100 flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:fixed lg:inset-y-0 bg-[#0B1221] border-r border-slate-800/80">
        <div className="px-5 py-5 border-b border-slate-800/80 flex items-center gap-3">
          <ShieldLogo className="w-9 h-9 shrink-0" />
          <div className="min-w-0">
            <div className="font-display text-sm font-bold text-slate-100 leading-tight truncate">
              Credit Card Fraud Shield
            </div>
            <div className="text-[11px] text-sky-400/90 mt-0.5 truncate">
              Intelligent Transaction Security
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto" aria-label="Primary Navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = page === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleNav(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${active ? 'text-sky-400' : 'text-slate-500'}`} />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {status && (
          <div className="px-4 py-4 border-t border-slate-800/80 text-[11px] text-slate-400 space-y-1.5">
            <div className="flex items-center justify-between">
              <span>Active Model</span>
              <span className="text-slate-200 font-medium">
                {status.settings.activeModel === 'tuned_xgboost'
                  ? 'Tuned XGBoost'
                  : status.settings.activeModel === 'decision_tree'
                  ? 'Decision Tree'
                  : 'Logistic Regression'}
              </span>
            </div>
            <div className="flex items-center justify-between font-mono tabular-nums">
              <span className="font-sans">Threshold</span>
              <span className="text-sky-400">{status.settings.threshold.toFixed(2)}</span>
            </div>
          </div>
        )}
      </aside>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-40 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/60"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative z-50 w-64 bg-[#0B1221] border-r border-slate-800 flex flex-col h-full">
            <div className="px-4 py-4 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <ShieldLogo className="w-8 h-8" />
                <span className="font-display text-sm font-bold text-slate-100">
                  Fraud Shield
                </span>
              </div>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                className="p-1.5 text-slate-400 hover:text-slate-100"
                aria-label="Close navigation menu"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <nav className="p-3 space-y-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = page === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleNav(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium ${
                      active
                        ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                        : 'text-slate-400 hover:text-slate-100'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Shell */}
      <div className="flex-1 lg:pl-64 flex flex-col min-w-0">
        <header className="sticky top-0 z-30 bg-[#080D1A]/90 backdrop-blur-md border-b border-slate-800/80 px-5 lg:px-8 py-3.5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              className="lg:hidden p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300"
              aria-label="Open navigation menu"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="min-w-0">
              <h1 className="font-display text-base lg:text-lg font-semibold text-slate-100 truncate">
                {activeMeta.label}
              </h1>
              <p className="text-xs text-slate-400 truncate hidden sm:block">
                {activeMeta.subtitle}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {page !== 'fraud-detection' && (
              <button
                type="button"
                onClick={() => handleNav('fraud-detection')}
                className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap"
              >
                Score Transaction
              </button>
            )}
            {page === 'fraud-detection' && (
              <button
                type="button"
                onClick={() => handleNav('settings')}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 transition-colors whitespace-nowrap"
              >
                <SlidersHorizontal className="w-3.5 h-3.5 text-sky-400" />
                Threshold Settings
              </button>
            )}
          </div>
        </header>

        {uploadProgress && (
          <div className="px-5 lg:px-8 pt-4">
            <div className="rounded-lg px-4 py-2.5 text-xs bg-sky-950/50 border border-sky-700/60 text-sky-200 flex items-center gap-3">
              <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse shrink-0" />
              <span className="font-mono">{uploadProgress}</span>
            </div>
          </div>
        )}

        {toast && (
          <div className="px-5 lg:px-8 pt-4">
            <div
              className={`rounded-lg px-4 py-2.5 text-xs flex items-center justify-between gap-3 border ${
                toast.type === 'success'
                  ? 'bg-emerald-950/40 border-emerald-700/60 text-emerald-200'
                  : 'bg-rose-950/40 border-rose-700/60 text-rose-200'
              }`}
            >
              <div className="flex items-center gap-2">
                {toast.type === 'success' && <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />}
                <span>{toast.text}</span>
              </div>
              <button
                type="button"
                onClick={() => setToast(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        <main className="flex-1 p-5 lg:p-8 max-w-[1400px] w-full mx-auto">
          {loading || !status ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-28 rounded-xl bg-slate-900/60 border border-slate-800 animate-pulse" />
              ))}
            </div>
          ) : (
            <>
              {page === 'dashboard' && (
                <DashboardPage
                  status={status}
                  onNavigate={handleNav}
                  onUploadCsv={handleUploadCsv}
                  onTrainBenchmark={() => handleTrainMode('benchmark')}
                  onTrainExistingKaggle={() => handleTrainMode('kaggle')}
                />
              )}
              {page === 'fraud-detection' && (
                <FraudDetectionPage
                  status={status}
                  onPredictionComplete={fetchStatus}
                />
              )}
              {page === 'transaction-history' && (
                <TransactionHistoryPage onHistoryMutated={fetchStatus} />
              )}
              {page === 'model-performance' && (
                <ModelPerformancePage
                  status={status}
                  onUploadCsv={handleUploadCsv}
                  onTrainBenchmark={() => handleTrainMode('benchmark')}
                  onTrainExistingKaggle={() => handleTrainMode('kaggle')}
                />
              )}
              {page === 'fraud-analytics' && (
                <FraudAnalyticsPage
                  status={status}
                  onUploadCsv={handleUploadCsv}
                  onTrainBenchmark={() => handleTrainMode('benchmark')}
                  onTrainExistingKaggle={() => handleTrainMode('kaggle')}
                />
              )}
              {page === 'settings' && (
                <SettingsPage
                  status={status}
                  onSettingsSaved={fetchStatus}
                  onUploadCsv={handleUploadCsv}
                  onTrainBenchmark={() => handleTrainMode('benchmark')}
                  onTrainExistingKaggle={() => handleTrainMode('kaggle')}
                />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
