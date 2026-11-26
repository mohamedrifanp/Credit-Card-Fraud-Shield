import React, { useState } from 'react';
import {
  CurvePoint,
  DBSCANPoint,
  ModelEvaluation,
  ModelId,
  ThresholdSweepPoint,
} from '../types';

const MODEL_COLORS: Record<ModelId, string> = {
  logistic_regression: '#94A3B8',
  decision_tree: '#F59E0B',
  tuned_xgboost: '#38BDF8',
};

export function FraudDistributionChart({
  legitimateCount,
  fraudCount,
}: {
  legitimateCount: number;
  fraudCount: number;
}) {
  const total = Math.max(1, legitimateCount + fraudCount);
  const legitPct = (legitimateCount / total) * 100;
  const fraudPct = (fraudCount / total) * 100;

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-300 font-medium">Legitimate Transactions (Class 0)</span>
          <span className="font-mono tabular-nums text-emerald-400">
            {legitimateCount.toLocaleString()} ({legitPct.toFixed(2)}%)
          </span>
        </div>
        <div className="h-3 w-full rounded bg-slate-900 overflow-hidden border border-slate-800">
          <div
            className="h-full bg-emerald-500/80 transition-all duration-300"
            style={{ width: `${Math.max(1, legitPct)}%` }}
          />
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-300 font-medium">Fraudulent Transactions (Class 1)</span>
          <span className="font-mono tabular-nums text-rose-400">
            {fraudCount.toLocaleString()} ({fraudPct.toFixed(4)}%)
          </span>
        </div>
        <div className="h-3 w-full rounded bg-slate-900 overflow-hidden border border-slate-800">
          <div
            className="h-full bg-rose-500 transition-all duration-300"
            style={{ width: `${Math.max(1.5, fraudPct)}%` }}
          />
        </div>
      </div>

      <p className="text-xs text-slate-400 leading-relaxed pt-1">
        Imbalance ratio:{' '}
        <span className="font-mono tabular-nums text-slate-200">
          {fraudCount > 0 ? Math.round(legitimateCount / fraudCount).toLocaleString() : 0}:1
        </span>{' '}
        legitimate-to-fraudulent transactions. Models use class-weighted training and PR-AUC
        optimization to prevent majority-class collapse.
      </p>
    </div>
  );
}

export function AmountHistogramChart({
  buckets,
}: {
  buckets: Array<{ bucket: string; legitimate: number; fraudulent: number }>;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  if (!buckets.length) {
    return <p className="text-xs text-slate-500 py-8 text-center">No transaction amount data loaded.</p>;
  }

  const totalLegit = buckets.reduce((s, b) => s + b.legitimate, 0) || 1;
  const totalFraud = buckets.reduce((s, b) => s + b.fraudulent, 0) || 1;

  // Normalize to class-internal share (%) so severe imbalance doesn't flatten fraud bars
  const normalized = buckets.map((b) => ({
    ...b,
    legitShare: (b.legitimate / totalLegit) * 100,
    fraudShare: (b.fraudulent / totalFraud) * 100,
  }));

  const maxShare = Math.max(
    10,
    ...normalized.map((b) => Math.max(b.legitShare, b.fraudShare))
  );

  const w = 520;
  const h = 210;
  const padL = 42;
  const padR = 12;
  const padT = 14;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const groupW = plotW / normalized.length;
  const barW = Math.max(10, groupW * 0.34);

  return (
    <div className="relative">
      <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
        <span>Class-Normalized Share by Transaction Amount (%)</span>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-emerald-500/80 inline-block" />
            Legitimate
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-rose-500 inline-block" />
            Fraudulent
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-52 select-none">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padT + plotH * (1 - t);
          return (
            <g key={t}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
              <text
                x={padL - 6}
                y={y + 3}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono"
              >
                {Math.round(maxShare * t)}%
              </text>
            </g>
          );
        })}

        {normalized.map((b, idx) => {
          const gx = padL + idx * groupW + groupW * 0.14;
          const lh = (b.legitShare / maxShare) * plotH;
          const fh = (b.fraudShare / maxShare) * plotH;
          const isHovered = hoverIdx === idx;
          return (
            <g
              key={b.bucket}
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
              className="cursor-pointer"
            >
              {isHovered && (
                <rect
                  x={padL + idx * groupW}
                  y={padT}
                  width={groupW}
                  height={plotH}
                  fill="#38BDF8"
                  fillOpacity="0.06"
                />
              )}
              <rect
                x={gx}
                y={padT + plotH - lh}
                width={barW}
                height={Math.max(2, lh)}
                rx="2"
                fill="#10B981"
                fillOpacity={isHovered ? 0.95 : 0.78}
              />
              <rect
                x={gx + barW + 3}
                y={padT + plotH - fh}
                width={barW}
                height={Math.max(2, fh)}
                rx="2"
                fill="#EF4444"
                fillOpacity={isHovered ? 1 : 0.85}
              />
              <text
                x={padL + idx * groupW + groupW / 2}
                y={h - 10}
                textAnchor="middle"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {b.bucket}
              </text>
            </g>
          );
        })}
      </svg>

      {hoverIdx !== null && normalized[hoverIdx] && (
        <div className="mt-1 flex items-center justify-between text-xs bg-slate-900/90 border border-slate-800 rounded-md px-3 py-1.5 font-mono tabular-nums">
          <span className="text-slate-200">{normalized[hoverIdx].bucket}</span>
          <span className="text-emerald-400">
            Legit: {normalized[hoverIdx].legitimate.toLocaleString()} ({normalized[hoverIdx].legitShare.toFixed(1)}%)
          </span>
          <span className="text-rose-400">
            Fraud: {normalized[hoverIdx].fraudulent.toLocaleString()} ({normalized[hoverIdx].fraudShare.toFixed(1)}%)
          </span>
        </div>
      )}
    </div>
  );
}

export function FraudTrendChart({
  trends,
}: {
  trends: Array<{ hourBucket: string; total: number; fraudulent: number; fraudRate: number; avgAmount: number }>;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  if (!trends.length) {
    return <p className="text-xs text-slate-500 py-8 text-center">No temporal trend data available.</p>;
  }

  const maxRate = Math.max(0.5, ...trends.map((t) => t.fraudRate));
  const w = 560;
  const h = 210;
  const padL = 44;
  const padR = 14;
  const padT = 14;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const coords = trends.map((t, i) => ({
    x: padL + (trends.length > 1 ? (i / (trends.length - 1)) * plotW : plotW / 2),
    y: padT + plotH - (t.fraudRate / maxRate) * plotH,
    ...t,
  }));

  const linePath = coords
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(' ');
  const areaPath =
    linePath +
    ` L ${coords[coords.length - 1].x.toFixed(1)} ${(padT + plotH).toFixed(1)} L ${coords[0].x.toFixed(1)} ${(padT + plotH).toFixed(1)} Z`;

  return (
    <div>
      <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
        <span>Fraud Rate (%) Across Transaction Time Window</span>
        <span className="font-mono tabular-nums text-sky-400">Peak: {maxRate.toFixed(2)}%</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-52 select-none">
        <defs>
          <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#38BDF8" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((r) => {
          const y = padT + plotH * (1 - r);
          return (
            <g key={r}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {(maxRate * r).toFixed(1)}%
              </text>
            </g>
          );
        })}

        <path d={areaPath} fill="url(#trendGrad)" />
        <path d={linePath} fill="none" stroke="#38BDF8" strokeWidth="2" />

        {coords.map((p, i) => (
          <g key={p.hourBucket}>
            {hoverIdx === i && (
              <line
                x1={p.x}
                y1={padT}
                x2={p.x}
                y2={padT + plotH}
                stroke="#38BDF8"
                strokeOpacity="0.45"
                strokeDasharray="2 2"
              />
            )}
            <circle
              cx={p.x}
              cy={p.y}
              r={hoverIdx === i ? 4.5 : 2.2}
              fill={p.fraudulent > 0 ? '#F43F5E' : '#38BDF8'}
              className="cursor-pointer"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
            />
            {(i === 0 || i === Math.floor(coords.length / 2) || i === coords.length - 1) && (
              <text x={p.x} y={h - 8} textAnchor="middle" className="fill-slate-400 text-[10px] font-mono">
                {p.hourBucket}
              </text>
            )}
          </g>
        ))}
      </svg>

      {hoverIdx !== null && coords[hoverIdx] && (
        <div className="mt-1 flex items-center justify-between text-xs bg-slate-900/90 border border-slate-800 rounded-md px-3 py-1.5 font-mono tabular-nums">
          <span className="text-slate-200">Hour {coords[hoverIdx].hourBucket}</span>
          <span className="text-slate-300">Tx: {coords[hoverIdx].total.toLocaleString()}</span>
          <span className="text-rose-400">Fraud: {coords[hoverIdx].fraudulent}</span>
          <span className="text-sky-400">Rate: {coords[hoverIdx].fraudRate.toFixed(2)}%</span>
        </div>
      )}
    </div>
  );
}

export function MultiModelCurveChart({
  title,
  xLabel,
  yLabel,
  models,
  curveKey,
  showDiagonal = false,
}: {
  title: string;
  xLabel: string;
  yLabel: string;
  models: Record<ModelId, ModelEvaluation>;
  curveKey: 'rocCurve' | 'prCurve';
  showDiagonal?: boolean;
}) {
  const [activeModel, setActiveModel] = useState<ModelId | 'all'>('all');
  const w = 440;
  const h = 250;
  const padL = 44;
  const padR = 16;
  const padT = 14;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const toPath = (pts: CurvePoint[]) =>
    pts
      .map((p, i) => {
        const px = padL + p.x * plotW;
        const py = padT + (1 - p.y) * plotH;
        return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
      })
      .join(' ');

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <span className="text-sm font-medium text-slate-200">{title}</span>
        <div className="flex items-center gap-3 text-xs">
          {(Object.keys(models) as ModelId[]).map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => setActiveModel(activeModel === id ? 'all' : id)}
              className={`flex items-center gap-1.5 px-2 py-0.5 rounded transition-opacity whitespace-nowrap ${
                activeModel === 'all' || activeModel === id ? 'opacity-100' : 'opacity-40'
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-xs inline-block"
                style={{ backgroundColor: MODEL_COLORS[id] }}
              />
              <span className="text-slate-300">{models[id].modelName}</span>
            </button>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-56 select-none">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padT + (1 - t) * plotH;
          const x = padL + t * plotW;
          return (
            <g key={t}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="#1E293B" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {t.toFixed(2)}
              </text>
              <text x={x} y={padT + plotH + 14} textAnchor="middle" className="fill-slate-500 text-[10px] font-mono">
                {t.toFixed(2)}
              </text>
            </g>
          );
        })}

        {showDiagonal && (
          <line
            x1={padL}
            y1={padT + plotH}
            x2={w - padR}
            y2={padT}
            stroke="#475569"
            strokeDasharray="4 4"
            strokeWidth="1.2"
          />
        )}

        {(Object.keys(models) as ModelId[]).map((id) => {
          const m = models[id];
          const dimmed = activeModel !== 'all' && activeModel !== id;
          return (
            <path
              key={id}
              d={toPath(m[curveKey])}
              fill="none"
              stroke={MODEL_COLORS[id]}
              strokeWidth={id === 'tuned_xgboost' ? 2.4 : 1.8}
              strokeOpacity={dimmed ? 0.2 : 0.95}
            />
          );
        })}

        <text
          x={padL + plotW / 2}
          y={h - 4}
          textAnchor="middle"
          className="fill-slate-400 text-[10px]"
        >
          {xLabel}
        </text>
        <text
          x={12}
          y={padT + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${padT + plotH / 2})`}
          className="fill-slate-400 text-[10px]"
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

export function ConfusionMatrixGrid({
  cm,
  modelName,
}: {
  cm: { tn: number; fp: number; fn: number; tp: number };
  modelName: string;
}) {
  const actualNeg = Math.max(1, cm.tn + cm.fp);
  const actualPos = Math.max(1, cm.fn + cm.tp);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>Held-Out Test Confusion Matrix — {modelName}</span>
        <span className="font-mono tabular-nums">
          Total: {(cm.tn + cm.fp + cm.fn + cm.tp).toLocaleString()}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div />
        <div className="text-center text-slate-400 font-medium py-1">Pred Legitimate</div>
        <div className="text-center text-slate-400 font-medium py-1">Pred Fraudulent</div>

        <div className="flex items-center justify-end pr-2 text-slate-400 font-medium">
          Actual Legitimate
        </div>
        <div className="rounded-lg bg-emerald-950/50 border border-emerald-800/60 p-3 text-center">
          <div className="text-[11px] text-emerald-400/80">True Negative</div>
          <div className="text-lg font-mono tabular-nums font-semibold text-emerald-300 mt-0.5">
            {cm.tn.toLocaleString()}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-slate-400">
            {((cm.tn / actualNeg) * 100).toFixed(2)}%
          </div>
        </div>
        <div className="rounded-lg bg-amber-950/35 border border-amber-800/50 p-3 text-center">
          <div className="text-[11px] text-amber-400/80">False Positive</div>
          <div className="text-lg font-mono tabular-nums font-semibold text-amber-300 mt-0.5">
            {cm.fp.toLocaleString()}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-slate-400">
            {((cm.fp / actualNeg) * 100).toFixed(2)}%
          </div>
        </div>

        <div className="flex items-center justify-end pr-2 text-slate-400 font-medium">
          Actual Fraudulent
        </div>
        <div className="rounded-lg bg-rose-950/40 border border-rose-800/60 p-3 text-center">
          <div className="text-[11px] text-rose-400/80">False Negative</div>
          <div className="text-lg font-mono tabular-nums font-semibold text-rose-300 mt-0.5">
            {cm.fn.toLocaleString()}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-slate-400">
            {((cm.fn / actualPos) * 100).toFixed(2)}%
          </div>
        </div>
        <div className="rounded-lg bg-sky-950/50 border border-sky-700/60 p-3 text-center">
          <div className="text-[11px] text-sky-400/90">True Positive</div>
          <div className="text-lg font-mono tabular-nums font-semibold text-sky-300 mt-0.5">
            {cm.tp.toLocaleString()}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-slate-400">
            {((cm.tp / actualPos) * 100).toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  );
}

export function FeatureImportanceChart({
  items,
}: {
  items: Array<{ feature: string; importance: number }>;
}) {
  const top = items.slice(0, 12);
  const maxImp = Math.max(0.01, ...top.map((i) => i.importance));

  return (
    <div className="space-y-2.5">
      {top.map((item) => {
        const widthPct = (item.importance / maxImp) * 100;
        return (
          <div key={item.feature} className="grid grid-cols-12 items-center gap-3 text-xs">
            <span className="col-span-2 font-mono text-slate-300 text-right">{item.feature}</span>
            <div className="col-span-8 h-2.5 rounded bg-slate-900 border border-slate-800 overflow-hidden">
              <div
                className="h-full bg-sky-500/85 rounded-r"
                style={{ width: `${Math.max(2, widthPct)}%` }}
              />
            </div>
            <span className="col-span-2 font-mono tabular-nums text-slate-400 text-right">
              {(item.importance * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function PcaVarianceChart({
  explainedRatio,
  cumulative,
  components95,
}: {
  explainedRatio: number[];
  cumulative: number[];
  components95: number;
}) {
  const w = 540;
  const h = 210;
  const padL = 42;
  const padR = 16;
  const padT = 14;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const n = explainedRatio.length;
  const barW = Math.max(4, (plotW / n) * 0.65);

  const cumPath = cumulative
    .map((v, i) => {
      const x = padL + (i + 0.5) * (plotW / n);
      const y = padT + (1 - v) * plotH;
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  const y95 = padT + (1 - 0.95) * plotH;

  return (
    <div>
      <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
        <span>Individual & Cumulative Explained Variance (PC1–PC{n})</span>
        <span className="font-mono tabular-nums text-sky-400">
          95% Threshold: {components95} components
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-52 select-none">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padT + (1 - t) * plotH;
          return (
            <g key={t}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {Math.round(t * 100)}%
              </text>
            </g>
          );
        })}

        <line x1={padL} y1={y95} x2={w - padR} y2={y95} stroke="#10B981" strokeDasharray="4 3" />

        {explainedRatio.map((val, idx) => {
          const x = padL + idx * (plotW / n) + (plotW / n - barW) / 2;
          const bh = Math.min(plotH, val * plotH * 3.5); // Scaled visually for individual components
          return (
            <rect
              key={idx}
              x={x}
              y={padT + plotH - bh}
              width={barW}
              height={Math.max(1.5, bh)}
              fill="#0EA5E9"
              fillOpacity={idx < components95 ? 0.75 : 0.25}
              rx="1"
            />
          );
        })}

        <path d={cumPath} fill="none" stroke="#38BDF8" strokeWidth="2" />
        <text x={padL + plotW / 2} y={h - 6} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Principal Component Index (1 to {n})
        </text>
      </svg>
    </div>
  );
}

export function DbscanScatterChart({ points }: { points: DBSCANPoint[] }) {
  const [hovered, setHovered] = useState<DBSCANPoint | null>(null);

  if (!points.length) {
    return <p className="text-xs text-slate-500 py-8 text-center">No DBSCAN points computed.</p>;
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);

  const w = 520;
  const h = 230;
  const pad = 28;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 mb-2">
        <span>DBSCAN Density Projection (PC1 vs PC2)</span>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-500 inline-block" />
            Core / Border Cluster
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" />
            Noise Outlier (Legit)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" />
            Fraud Label
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-56 bg-[#0B1120] rounded-lg border border-slate-800/80">
        <line x1={pad} y1={h / 2} x2={w - pad} y2={h / 2} stroke="#1E293B" />
        <line x1={w / 2} y1={pad} x2={w / 2} y2={h - pad} stroke="#1E293B" />
        {points.map((p, idx) => {
          const cx = pad + ((p.x - minX) / spanX) * plotW;
          const cy = h - pad - ((p.y - minY) / spanY) * plotH;
          const color =
            p.actualClass === 1
              ? '#EF4444'
              : p.isNoise
              ? '#F59E0B'
              : '#64748B';
          return (
            <circle
              key={idx}
              cx={cx}
              cy={cy}
              r={p.actualClass === 1 ? 3.5 : p.isNoise ? 2.8 : 2}
              fill={color}
              fillOpacity={p.actualClass === 1 || p.isNoise ? 0.9 : 0.45}
              className="cursor-pointer"
              onMouseEnter={() => setHovered(p)}
              onMouseLeave={() => setHovered(null)}
            />
          );
        })}
      </svg>

      {hovered && (
        <div className="mt-1.5 flex items-center justify-between text-xs bg-slate-900/90 border border-slate-800 rounded-md px-3 py-1.5 font-mono tabular-nums">
          <span className="text-slate-300">
            {hovered.isNoise ? 'Cluster: -1 (Noise Outlier)' : `Cluster: #${hovered.cluster}`}
          </span>
          <span className="text-slate-300">Amount: ${hovered.amount.toFixed(2)}</span>
          <span className={hovered.actualClass === 1 ? 'text-rose-400' : 'text-emerald-400'}>
            Actual Label: {hovered.actualClass === 1 ? 'Fraud (1)' : 'Legitimate (0)'}
          </span>
        </div>
      )}
    </div>
  );
}

export function ThresholdImpactChart({
  sweep,
  activeThreshold,
}: {
  sweep: ThresholdSweepPoint[];
  activeThreshold: number;
}) {
  if (!sweep.length) return null;
  const w = 500;
  const h = 210;
  const padL = 42;
  const padR = 16;
  const padT = 16;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const toX = (t: number) => padL + ((t - 0.1) / 0.85) * plotW;
  const toY = (v: number) => padT + (1 - v) * plotH;

  const buildPath = (key: 'precision' | 'recall' | 'f1') =>
    sweep
      .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${toX(pt.threshold).toFixed(1)} ${toY(pt[key]).toFixed(1)}`)
      .join(' ');

  const markerX = toX(Math.min(0.95, Math.max(0.1, activeThreshold)));

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 mb-2">
        <span>Threshold Trade-Off (Precision vs Recall vs F1)</span>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-sky-400">
            <span className="w-2.5 h-0.5 bg-sky-400 inline-block" /> Precision
          </span>
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2.5 h-0.5 bg-emerald-400 inline-block" /> Recall
          </span>
          <span className="flex items-center gap-1.5 text-amber-400">
            <span className="w-2.5 h-0.5 bg-amber-400 inline-block" /> F1-Score
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-52 select-none">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => {
          const y = toY(v);
          return (
            <g key={v}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {Math.round(v * 100)}%
              </text>
            </g>
          );
        })}

        <path d={buildPath('precision')} fill="none" stroke="#38BDF8" strokeWidth="2" />
        <path d={buildPath('recall')} fill="none" stroke="#10B981" strokeWidth="2" />
        <path d={buildPath('f1')} fill="none" stroke="#F59E0B" strokeWidth="2" strokeDasharray="4 2" />

        <line x1={markerX} y1={padT} x2={markerX} y2={padT + plotH} stroke="#E2E8F0" strokeWidth="1.5" />
        <text x={markerX} y={padT - 4} textAnchor="middle" className="fill-slate-200 text-[10px] font-mono">
          {activeThreshold.toFixed(2)}
        </text>

        {[0.1, 0.3, 0.5, 0.7, 0.9].map((t) => (
          <text
            key={t}
            x={toX(t)}
            y={padT + plotH + 16}
            textAnchor="middle"
            className="fill-slate-500 text-[10px] font-mono"
          >
            {t.toFixed(1)}
          </text>
        ))}
      </svg>
    </div>
  );
}
