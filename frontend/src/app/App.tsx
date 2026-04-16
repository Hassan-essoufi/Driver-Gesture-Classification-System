import { useState } from 'react';
import {
  Upload, Activity, AlertTriangle, CheckCircle2, XCircle,
  BarChart3, Image as ImageIcon, Loader2, Clock, Cpu,
  TrendingUp, List, Home, ChevronRight, Zap
} from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

type ModelName = 'resnet50' | 'efficientnet_b3';
type RiskLevel = 'safe' | 'low' | 'medium' | 'high';
type Page = 'dashboard' | 'analyze' | 'history';

type Result = {
  id: string;
  timestamp: Date;
  class: string;
  classIndex: number;
  confidence: number;
  riskLevel: RiskLevel;
  imageUrl: string;
  processingTime: number;
  model: ModelName;
};

const CLASSES = [
  'Safe Driving', 'Texting Right', 'Phone Right',
  'Texting Left', 'Phone Left', 'Operating Radio',
  'Drinking', 'Reaching Behind', 'Hair & Makeup', 'Talking to Passenger',
];

const RISK: RiskLevel[] = [
  'safe', 'high', 'high', 'high', 'high',
  'medium', 'medium', 'medium', 'low', 'low',
];

const RISK_COLORS: Record<RiskLevel, string> = {
  safe: '#22c55e', low: '#eab308', medium: '#f97316', high: '#ef4444',
};

const PIE_COLORS = ['#22c55e', '#ef4444', '#ef4444', '#ef4444', '#ef4444', '#f97316', '#f97316', '#f97316', '#eab308', '#eab308'];

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const [history, setHistory] = useState<Result[]>([]);
  const [drag, setDrag] = useState(false);
  const [model, setModel] = useState<ModelName>('efficientnet_b3');
  const [error, setError] = useState<string | null>(null);

  const stats = {
    total: history.length,
    safe: history.filter(r => r.classIndex === 0).length,
    distracted: history.filter(r => r.classIndex !== 0).length,
    avgConf: history.length ? history.reduce((s, r) => s + r.confidence, 0) / history.length : 0,
    avgTime: history.length ? history.reduce((s, r) => s + r.processingTime, 0) / history.length : 0,
  };

  const classDist = CLASSES.map((name, i) => ({
    name: name.length > 12 ? name.slice(0, 12) + '…' : name,
    full: name,
    count: history.filter(r => r.classIndex === i).length,
    color: PIE_COLORS[i],
  })).filter(d => d.count > 0);

  const trendData = history.slice(-10).reverse().map((r, i) => ({
    n: i + 1,
    confidence: parseFloat((r.confidence * 100).toFixed(1)),
    risk: r.riskLevel === 'safe' ? 0 : r.riskLevel === 'low' ? 1 : r.riskLevel === 'medium' ? 2 : 3,
  }));

  const pickFile = (f: File) => {
    if (!f.type.startsWith('image/')) return;
    setFile(f);
    const reader = new FileReader();
    reader.onload = e => {
      setImage(e.target?.result as string);
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(f);
  };

  const runAnalysis = async () => {
    if (!file || !image) return;
    setProcessing(true);
    setProgress(0);
    setError(null);

    const steps = ['Uploading image…', 'Preprocessing…', `Running ${model}…`, 'Analyzing…', 'Done'];
    let si = 0;
    const iv = setInterval(() => {
      if (si < steps.length - 1) {
        setProcessingStep(steps[si]);
        setProgress(Math.round(((si + 1) / steps.length) * 85));
        si++;
      }
    }, 350);

    const t0 = Date.now();
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_URL}/predict?model_name=${model}`, { method: 'POST', body: fd });
      clearInterval(iv);
      if (!res.ok) {
        const e = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(e.detail ?? 'Prediction failed');
      }
      const data: { class: string; label_id: number; confidence: number } = await res.json();
      setProgress(100);
      setProcessingStep('Done');
      const r: Result = {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        class: data.class === 'uncertain' ? 'Uncertain' : (CLASSES[data.label_id] ?? data.class),
        classIndex: data.label_id,
        confidence: data.confidence,
        riskLevel: RISK[data.label_id] ?? 'medium',
        imageUrl: image,
        processingTime: Date.now() - t0,
        model,
      };
      setResult(r);
      setHistory(prev => [r, ...prev]);
    } catch (e: unknown) {
      clearInterval(iv);
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setProcessing(false);
    }
  };

  const riskBadge = (risk: RiskLevel) => {
    const styles: Record<RiskLevel, string> = {
      safe: 'bg-green-500/15 text-green-400 border-green-500/30',
      low: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
      medium: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
      high: 'bg-red-500/15 text-red-400 border-red-500/30',
    };
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs font-semibold uppercase tracking-wide ${styles[risk]}`}>
        {risk === 'safe' ? <CheckCircle2 size={10} /> : risk === 'high' ? <XCircle size={10} /> : <AlertTriangle size={10} />}
        {risk}
      </span>
    );
  };

  // ── Sidebar ──────────────────────────────────────────────
  const nav = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'analyze', label: 'Analyze', icon: Cpu },
    { id: 'history', label: 'History', icon: List },
  ] as const;

  return (
    <div className="flex h-screen bg-[#0a0a0f] text-slate-100 overflow-hidden">

      {/* ── Sidebar ── */}
      <aside className="w-60 flex-shrink-0 flex flex-col bg-[#0d0d14] border-r border-slate-800">
        {/* Spacer to align nav below header */}
        <div className="px-5 py-5" />

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-1">
          {nav.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                page === id
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                  : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
              }`}
            >
              <Icon size={16} />
              {label}
              {page === id && <ChevronRight size={12} className="ml-auto" />}
            </button>
          ))}
        </nav>

        {/* Model selector */}
        <div className="p-4 border-t border-slate-800">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">Active Model</p>
          <select
            value={model}
            onChange={e => setModel(e.target.value as ModelName)}
            className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 cursor-pointer focus:outline-none focus:border-blue-500"
          >
            <option value="efficientnet_b3">EfficientNet-B3</option>
            <option value="resnet50">ResNet50</option>
          </select>
          <p className="text-[10px] text-slate-600 mt-1.5">
            {model === 'efficientnet_b3' ? '99.15% test acc' : '99.02% test acc'}
          </p>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-4 bg-[#0d0d14]/50">
          <div>
            <h1 className="text-lg font-semibold text-white capitalize">{page}</h1>
            <p className="text-xs text-slate-500">
              {page === 'dashboard' && 'Overview of detection activity'}
              {page === 'analyze' && 'Upload an image to classify driver behavior'}
              {page === 'history' && `${history.length} prediction${history.length !== 1 ? 's' : ''} recorded`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-xl">
              <Cpu size={22} className="text-white" />
            </div>
            <p className="text-xl font-bold text-white tracking-wide">DriverGuard</p>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-6">

          {/* ══════════ DASHBOARD ══════════ */}
          {page === 'dashboard' && (
            <div className="space-y-6">
              {/* Stat cards */}
              <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                {[
                  { label: 'Total Analyzed', value: stats.total, icon: BarChart3, color: 'text-blue-400', bg: 'bg-blue-500/10 border-blue-500/20' },
                  { label: 'Safe Driving', value: stats.safe, icon: CheckCircle2, color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/20' },
                  { label: 'Distracted', value: stats.distracted, icon: AlertTriangle, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20' },
                  { label: 'Avg Confidence', value: stats.total ? `${(stats.avgConf * 100).toFixed(1)}%` : '—', icon: TrendingUp, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20' },
                  { label: 'Avg Response', value: stats.total ? `${stats.avgTime.toFixed(0)}ms` : '—', icon: Zap, color: 'text-purple-400', bg: 'bg-purple-500/10 border-purple-500/20' },
                ].map(({ label, value, icon: Icon, color, bg }) => (
                  <div key={label} className={`p-4 rounded-xl border ${bg}`}>
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-xs text-slate-500">{label}</p>
                      <Icon size={16} className={color} />
                    </div>
                    <p className={`text-2xl font-bold ${color}`}>{value}</p>
                  </div>
                ))}
              </div>

              {/* Charts row */}
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Confidence trend */}
                <div className="bg-[#0d0d14] border border-slate-800 rounded-xl p-5">
                  <h3 className="text-sm font-semibold text-slate-300 mb-4">Confidence Trend</h3>
                  {trendData.length === 0 ? (
                    <div className="h-48 flex items-center justify-center text-slate-600 text-sm">No data yet</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={180}>
                      <AreaChart data={trendData}>
                        <defs>
                          <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="n" stroke="#475569" tick={{ fontSize: 10 }} />
                        <YAxis domain={[0, 100]} stroke="#475569" tick={{ fontSize: 10 }} unit="%" />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0d0d14', border: '1px solid #1e293b', borderRadius: 8, fontSize: 12 }}
                          formatter={(v: number) => [`${v}%`, 'Confidence']}
                        />
                        <Area type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2} fill="url(#cg)" dot={{ fill: '#3b82f6', r: 3 }} />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>

                {/* Class distribution */}
                <div className="bg-[#0d0d14] border border-slate-800 rounded-xl p-5">
                  <h3 className="text-sm font-semibold text-slate-300 mb-4">Class Distribution</h3>
                  {classDist.length === 0 ? (
                    <div className="h-48 flex items-center justify-center text-slate-600 text-sm">No data yet</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart data={classDist} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                        <XAxis type="number" stroke="#475569" tick={{ fontSize: 10 }} allowDecimals={false} />
                        <YAxis type="category" dataKey="name" stroke="#475569" tick={{ fontSize: 10 }} width={80} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0d0d14', border: '1px solid #1e293b', borderRadius: 8, fontSize: 12 }}
                          formatter={(v: number, _: string, p: { payload?: { full?: string } }) => [v, p.payload?.full ?? '']}
                        />
                        <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                          {classDist.map((d, i) => <Cell key={i} fill={d.color} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              {/* Recent predictions */}
              <div className="bg-[#0d0d14] border border-slate-800 rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-slate-300">Recent Predictions</h3>
                  {history.length > 0 && (
                    <button onClick={() => setPage('history')} className="text-xs text-blue-400 hover:text-blue-300">View all →</button>
                  )}
                </div>
                {history.length === 0 ? (
                  <div className="py-10 text-center text-slate-600 text-sm">
                    <ImageIcon size={32} className="mx-auto mb-2 opacity-30" />
                    No predictions yet — go to Analyze
                  </div>
                ) : (
                  <div className="space-y-2">
                    {history.slice(0, 5).map(r => (
                      <div key={r.id} className="flex items-center gap-4 p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 transition-colors">
                        <img src={r.imageUrl} alt="" className="size-10 rounded-lg object-cover border border-slate-700 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-slate-200 truncate">{r.class}</p>
                          <p className="text-xs text-slate-500">{r.model} · {r.timestamp.toLocaleTimeString()}</p>
                        </div>
                        <div className="text-right flex-shrink-0">
                          {riskBadge(r.riskLevel)}
                          <p className="text-xs text-slate-500 mt-1">{(r.confidence * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══════════ ANALYZE ══════════ */}
          {page === 'analyze' && (
            <div className="h-full grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 113px)' }}>

              {/* Left — Upload */}
              <div className="bg-[#0d0d14] border border-slate-800 rounded-xl p-5 flex flex-col gap-4 overflow-hidden">
                <h2 className="text-sm font-semibold text-slate-300 flex-shrink-0">Upload Image</h2>

                {/* Drop zone */}
                <div
                  onDragOver={e => { e.preventDefault(); setDrag(true); }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) pickFile(f); }}
                  className={`relative border-2 border-dashed rounded-xl text-center transition-all cursor-pointer flex-1 flex flex-col items-center justify-center ${
                    drag ? 'border-blue-500 bg-blue-500/5'
                    : image ? 'border-slate-700 bg-slate-800/20 p-3'
                    : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/20'
                  }`}
                >
                  <input
                    type="file"
                    accept="image/*"
                    onChange={e => e.target.files?.[0] && pickFile(e.target.files[0])}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  {image ? (
                    <div className="flex flex-col items-center gap-2 w-full h-full">
                      <img src={image} alt="preview" className="rounded-lg object-contain border border-slate-700 w-full flex-1 min-h-0" />
                      <p className="text-xs text-slate-500 flex-shrink-0">{file?.name} · Click or drop to change</p>
                    </div>
                  ) : (
                    <>
                      <Upload size={32} className="mb-3 text-slate-600" />
                      <p className="text-slate-300 font-medium mb-1">Drop image here or click to browse</p>
                      <p className="text-xs text-slate-500">JPG, PNG, WebP · Max 5MB</p>
                    </>
                  )}
                </div>

                {/* Button + progress */}
                <div className="flex-shrink-0 space-y-3">
                  <button
                    onClick={runAnalysis}
                    disabled={!image || processing}
                    className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 text-sm"
                  >
                    {processing ? <><Loader2 size={15} className="animate-spin" />{processingStep}</> : <><Zap size={15} />Run Analysis</>}
                  </button>

                  {processing && (
                    <div>
                      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
                      </div>
                      <p className="text-xs text-slate-500 mt-1 text-right">{progress}%</p>
                    </div>
                  )}

                  {error && (
                    <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-sm text-red-400">
                      <XCircle size={15} className="flex-shrink-0" />{error}
                    </div>
                  )}
                </div>

              </div>

              {/* Right — Result */}
              <div className="bg-[#0d0d14] border border-slate-800 rounded-xl p-5 flex flex-col gap-4 overflow-hidden">
                <h2 className="text-sm font-semibold text-slate-300 flex-shrink-0">Analysis Result</h2>

                {!result && !processing ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
                    <ImageIcon size={40} className="mb-3 opacity-30" />
                    <p className="text-sm">Upload an image and run analysis</p>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col gap-4 min-h-0">
                    {/* Result image */}
                    {result && (
                      <div className="flex-1 min-h-0 rounded-xl overflow-hidden border border-slate-700">
                        <img src={result.imageUrl} alt="result" className="w-full h-full object-contain bg-slate-900" />
                      </div>
                    )}

                    {/* Cards */}
                    {result && !processing && (
                      <div className="flex-shrink-0 grid grid-cols-2 gap-3">
                        {/* Class */}
                        <div className="p-3 rounded-xl border border-slate-700 bg-slate-800/30 col-span-2">
                          <p className="text-xs text-slate-500 mb-0.5">Detected Behavior</p>
                          <p className="text-lg font-bold text-white">{result.class}</p>
                          <p className="text-xs text-slate-500">C{result.classIndex} · {result.model}</p>
                        </div>

                        {/* Confidence */}
                        <div className="p-3 rounded-xl border border-slate-700 bg-slate-800/30">
                          <p className="text-xs text-slate-500 mb-1">Confidence</p>
                          <p className="text-xl font-bold text-blue-400">{(result.confidence * 100).toFixed(2)}%</p>
                          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden mt-2">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${result.confidence * 100}%` }} />
                          </div>
                        </div>

                        {/* Risk */}
                        <div className="p-3 rounded-xl border border-slate-700 bg-slate-800/30">
                          <p className="text-xs text-slate-500 mb-1">Risk Level</p>
                          <div className="mt-1">{riskBadge(result.riskLevel)}</div>
                          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden mt-2">
                            <div className="h-full rounded-full" style={{
                              width: result.riskLevel === 'safe' ? '10%' : result.riskLevel === 'low' ? '35%' : result.riskLevel === 'medium' ? '65%' : '100%',
                              backgroundColor: RISK_COLORS[result.riskLevel],
                            }} />
                          </div>
                        </div>

                        {/* Time */}
                        <div className="p-3 rounded-xl border border-slate-700 bg-slate-800/30">
                          <div className="flex items-center gap-1 text-slate-500 mb-1"><Clock size={11} /><p className="text-xs">Response</p></div>
                          <p className="text-sm font-semibold text-slate-200">{result.processingTime}ms</p>
                        </div>

                        {/* Model */}
                        <div className="p-3 rounded-xl border border-slate-700 bg-slate-800/30">
                          <div className="flex items-center gap-1 text-slate-500 mb-1"><Activity size={11} /><p className="text-xs">Model</p></div>
                          <p className="text-sm font-semibold text-slate-200 truncate">{result.model}</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══════════ HISTORY ══════════ */}
          {page === 'history' && (
            <div className="space-y-4">
              {/* Summary row */}
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: 'Total', value: stats.total, color: 'text-slate-200' },
                  { label: 'Safe', value: stats.safe, color: 'text-green-400' },
                  { label: 'Distracted', value: stats.distracted, color: 'text-red-400' },
                ].map(s => (
                  <div key={s.label} className="bg-[#0d0d14] border border-slate-800 rounded-xl p-4">
                    <p className="text-xs text-slate-500 mb-1">{s.label}</p>
                    <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
                  </div>
                ))}
              </div>

              {/* Table */}
              <div className="bg-[#0d0d14] border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-300">All Predictions</h3>
                  {history.length > 0 && (
                    <button
                      onClick={() => { setHistory([]); setResult(null); }}
                      className="text-xs text-slate-500 hover:text-red-400 transition-colors"
                    >
                      Clear all
                    </button>
                  )}
                </div>

                {history.length === 0 ? (
                  <div className="py-16 text-center text-slate-600">
                    <List size={32} className="mx-auto mb-2 opacity-30" />
                    <p className="text-sm">No predictions yet</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-800 text-xs text-slate-500 uppercase tracking-wide">
                          <th className="px-5 py-3 text-left">Image</th>
                          <th className="px-5 py-3 text-left">Class</th>
                          <th className="px-5 py-3 text-left">Risk</th>
                          <th className="px-5 py-3 text-left">Confidence</th>
                          <th className="px-5 py-3 text-left">Model</th>
                          <th className="px-5 py-3 text-left">Time</th>
                          <th className="px-5 py-3 text-left">Response</th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.map((r, i) => (
                          <tr
                            key={r.id}
                            className={`border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors ${i % 2 === 0 ? '' : 'bg-slate-800/10'}`}
                          >
                            <td className="px-5 py-3">
                              <img src={r.imageUrl} alt="" className="size-10 rounded-lg object-cover border border-slate-700" />
                            </td>
                            <td className="px-5 py-3">
                              <p className="text-slate-200 font-medium">{r.class}</p>
                              <p className="text-xs text-slate-500">C{r.classIndex}</p>
                            </td>
                            <td className="px-5 py-3">{riskBadge(r.riskLevel)}</td>
                            <td className="px-5 py-3">
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${r.confidence * 100}%` }} />
                                </div>
                                <span className="text-slate-300 text-xs">{(r.confidence * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                            <td className="px-5 py-3 text-xs text-slate-400">{r.model}</td>
                            <td className="px-5 py-3 text-xs text-slate-400">{r.timestamp.toLocaleTimeString()}</td>
                            <td className="px-5 py-3 text-xs text-slate-400">{r.processingTime}ms</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
