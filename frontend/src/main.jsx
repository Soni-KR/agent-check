import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ShieldCheck, FlaskConical, ArrowUpRight, Play, Check, X, ChevronRight, Clock3, Layers3, CircleHelp, LoaderCircle, Download, RotateCcw, LockKeyhole, Terminal, Activity, Bot } from 'lucide-react';
import AgentWorkspace, { ActionTrace } from './AgentWorkspace';
import './style.css';

const agentNames = { demo_safe: 'Guarded demo agent', demo_unsafe: 'Vulnerable demo agent', llm: 'Real LLM', autonomous: 'Autonomous tool agent' };
const icons = { math: Activity, format: Terminal, secret: LockKeyhole, injection: ShieldCheck };
const date = value => new Date(value.endsWith('Z') || /[+-]\d\d:\d\d$/.test(value) ? value : value + 'Z').toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

async function api(path, options) {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Request failed. Check that the backend is running.');
  }
  return response.json();
}

function App() {
  const [health, setHealth] = useState(null);
  const [cases, setCases] = useState([]);
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [agent, setAgent] = useState('autonomous');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(null);
  const [tab, setTab] = useState('agent');
  const [help, setHelp] = useState(false);

  async function load() {
    try {
      const [h, c, r] = await Promise.all([api('/health'), api('/cases'), api('/runs')]);
      setHealth(h); setCases(c); setRuns(r);
      setSelected(current => current || r[0] || null); setError('');
    } catch (err) { setError(err.message); }
  }
  useEffect(() => { load(); }, []);

  async function runSuite() {
    setBusy(true); setError(''); setExpanded(null);
    try {
      const report = await api('/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ agent }) });
      setSelected(report); setRuns(previous => [report, ...previous].slice(0, 50)); setTab('lab');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(selected, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = `agent-check-${selected.id}.json`; a.click(); URL.revokeObjectURL(url);
  }

  const results = selected?.results || cases;
  return <div className="app">
    <aside className="sidebar">
      <a href="#" className="brand" onClick={() => setTab('lab')}><span className="brand-icon"><ShieldCheck size={23} /></span>agent<span className="brand-light">check</span><span className="brand-dot">.</span></a>
      <div className="workspace"><span className="avatar">AC</span><div>Personal workspace<small>Local project</small></div><Layers3 size={15} /></div>
      <p className="nav-label">WORKSPACE</p>
      <nav aria-label="Main navigation">
        <button className={tab === 'agent' ? 'nav active' : 'nav'} onClick={() => setTab('agent')}><Bot size={18} /> Autonomous agent</button>
        <button className={tab === 'lab' ? 'nav active' : 'nav'} onClick={() => setTab('lab')}><FlaskConical size={18} /> Evaluation lab</button>
        <button className={tab === 'history' ? 'nav active' : 'nav'} onClick={() => setTab('history')}><Clock3 size={18} /> Run history <span className="count">{runs.length}</span></button>
      </nav>
      <div className="sidebar-bottom"><div className="stack-label">SMALL STACK. REAL CHECKS.</div><div className="stack-tags"><span>Python</span><span>FastAPI</span><span>React</span><span>SQL</span></div><button className="help-button" onClick={() => setHelp(!help)}><CircleHelp size={17} /> How it works <ArrowUpRight size={15} /></button></div>
    </aside>

    <div className="main-shell">
      <header className="topbar"><span>Workspace <ChevronRight size={13} /> <b>{tab === 'agent' ? 'Autonomous agent' : tab === 'lab' ? 'Evaluation lab' : 'Run history'}</b></span><span className="connection"><i className={health ? '' : 'offline'} /> {health ? 'API connected' : 'Connecting to API'}<span className="avatar small">M</span></span></header>
      <main>
        <div className="page-heading"><div><div className="eyebrow">AI AGENT WORKSPACE / V2.0</div><h1>{tab === 'agent' ? 'One goal. Let it work.' : tab === 'lab' ? 'Test before you trust.' : 'Every run, in one place.'}</h1><p>{tab === 'agent' ? 'A model that chooses tools, uses their results, and remembers.' : tab === 'lab' ? 'A little more confidence before your next deployment.' : 'Revisit your last 50 evaluations and inspect the results.'}</p></div><span className="local-badge"><span /> LOCAL ENVIRONMENT</span></div>

        {help && <section className="help-panel"><h3>A tiny testing pipeline</h3><p>Choose an agent, run four fixed checks, and inspect its actual answers. Every check must pass for the demo gate to open. The autonomous agent chooses from four tools and uses their results to complete goals. Choose it in the lab to test the same agent loop. Demo agents use simulated responses; Real LLM is a single model call. History is saved in {health?.database === 'postgresql' ? 'PostgreSQL' : 'SQLite'}.</p><p>Agent tasks and persistent notes are in the Autonomous agent tab. The action trace shows actual tool calls and observations. This project reports a gate decision; it does not deploy software.</p><button onClick={() => setHelp(false)}>Got it <Check size={14} /></button></section>}
        {error && <div className="error" role="alert"><span>{error}</span><button onClick={load}><RotateCcw size={14} /> Retry connection</button></div>}

        {tab === 'agent' ? <AgentWorkspace api={api} health={health}/> : tab === 'lab' ? <>
          <section className="runner card"><div className="runner-icon"><FlaskConical size={23} /></div><div className="runner-copy"><h2>Start an evaluation</h2><p>One agent. Four checks. A clear result.</p></div><div className="run-controls"><label htmlFor="agent">Agent to evaluate</label><div className="control-row"><select id="agent" value={agent} onChange={e => setAgent(e.target.value)} disabled={busy}><option value="demo_safe">Guarded demo agent</option><option value="demo_unsafe">Vulnerable demo agent</option><option value="autonomous" disabled={!health?.llm_configured}>Autonomous tool agent{!health?.llm_configured ? " · model not ready" : ""}</option><option value="llm" disabled={!health?.llm_configured}>Real LLM{!health?.llm_configured ? ' · model not ready' : ` · ${health.model}`}</option></select><button className="primary" onClick={runSuite} disabled={busy || !health || ((agent === "autonomous" || agent === "llm") && !health.llm_configured)}>{busy ? <LoaderCircle size={16} className="spin" /> : <Play size={15} fill="currentColor" />}{busy ? 'Running checks…' : 'Run evaluation'}</button></div></div></section>

          <div className="metrics"><div className="metric card"><div className="metric-label">Checks passed <Check size={16} /></div><div className="metric-value">{selected ? selected.passed : '—'}<span>/ {cases.length || 4}</span></div><small>{selected ? 'Across reliability and safety' : 'Your results will appear here'}</small></div><div className="metric card"><div className="metric-label">Pass rate <Activity size={16} /></div><div className="metric-value">{selected ? selected.score : '—'}<span>%</span></div><div className="progress"><div style={{ width: `${selected?.score || 0}%` }} /></div></div><div className={`metric card gate ${selected ? (selected.deployment_allowed ? 'good' : 'bad') : ''}`}><div className="metric-label">Deployment gate <ShieldCheck size={17} /></div><div className="gate-value">{selected ? (selected.deployment_allowed ? <><span className="status-icon"><Check size={16} /></span> Pass</> : <><span className="status-icon"><X size={16} /></span> Blocked</>) : 'Awaiting evaluation'}</div><small>{selected ? (selected.deployment_allowed ? 'All checks passed in this suite' : 'Resolve failed checks before proceeding') : 'All four checks must pass'}</small></div></div>

          <section className="suite card"><div className="section-heading"><div><h2>Core test suite <span className="pill">4 checks</span></h2><p>{selected ? `${agentNames[selected.agent]} · ${date(selected.created_at)}` : 'Simple checks for common failure modes.'}</p></div>{selected && <button className="text-button" onClick={download}><Download size={15} /> Export JSON</button>}</div><div className="table-head"><span>TEST CASE</span><span>CATEGORY</span><span>RESULT</span></div><div className="test-list">{results.map((test, index) => { const Icon = icons[test.id] || ShieldCheck; const open = expanded === test.id; return <div className="test-item" key={test.id}><button className="test-row" aria-expanded={open} onClick={() => setExpanded(open ? null : test.id)}><div className="test-name"><span className="test-number">0{index + 1}</span><span className="test-icon"><Icon size={18} /></span><span>{test.name}</span></div><span className={`category ${test.category.toLowerCase()}`}>{test.category}</span><span className="result-cell">{selected ? <span className={`result ${test.passed ? 'pass' : 'fail'}`}>{test.passed ? <Check size={13} /> : <X size={13} />}{test.passed ? 'Passed' : 'Failed'}</span> : <span className="pending">Ready</span>}<ChevronRight size={15} className={open ? 'rotated' : ''} /></span></button>{open && <div className="test-detail"><div><b>Prompt</b><p>{test.prompt}</p></div><div><b>Expected</b><code>{test.expected}</code></div>{selected && <><div><b>Agent response</b><code>{test.error || test.output || '(empty response)'}</code></div><small>{test.duration_ms} ms · Strict text match{selected.agent === "autonomous" && test.id === "math" ? " + verified calculator use" : ""}</small>{test.trace?.length > 0 && <ActionTrace entries={test.trace}/>}</>}</div>}</div>; })}</div><div className="suite-footer"><LockKeyhole size={14} /><span>Test secrets are fake. Results are saved locally.</span><span className="suite-version">suite / {selected?.suite_version || "2.0"}</span></div></section>

          <div className="bottom-note"><CircleHelp size={17} /><p>A starting point, not a safety guarantee. Add realistic test cases for your agent before using this gate in production.</p><button onClick={() => setHelp(!help)}>How it works <ArrowUpRight size={14} /></button></div>
        </> : <section className="card history"><div className="section-heading"><div><h2>Evaluation history</h2><p>Stored in {health?.database === 'postgresql' ? 'PostgreSQL' : 'SQLite'} · newest first</p></div><button className="text-button" onClick={load}><RotateCcw size={15} /> Refresh</button></div>{runs.length ? runs.map(run => <button className="history-row" key={run.id} onClick={() => { setSelected(run); setExpanded(null); setTab('lab'); }}><div><b>{agentNames[run.agent]}</b><small>{date(run.created_at)} · {run.id.slice(0, 8)}</small></div><span>{run.passed}/{run.total} passed</span><span className={`result ${run.deployment_allowed ? 'pass' : 'fail'}`}>{run.deployment_allowed ? 'Pass' : 'Blocked'}</span><ChevronRight size={16} /></button>) : <div className="empty"><Clock3 size={30} /><h3>Your first run starts here.</h3><p>Run an evaluation to build your test history.</p><button className="primary" onClick={() => setTab('lab')}>Open evaluation lab <ChevronRight size={15} /></button></div>}</section>}
        <footer className="page-footer"><span>Agent Check <span> / </span> A small lab for more reliable AI.</span><span>FastAPI + React + {health?.database === 'postgresql' ? 'PostgreSQL' : 'SQLite'}</span></footer>
      </main>
    </div>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
