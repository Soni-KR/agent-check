import React, { useEffect, useState } from 'react';
import { Bot, Play, LoaderCircle, RefreshCw, ArrowUpRight, Calculator, NotebookPen, Search, Check, X, ChevronRight } from 'lucide-react';

export function ActionTrace({ entries = [] }) {
  if (!entries.length) return <p className="trace-empty">No actions recorded.</p>;
  return <ol className="action-trace">{entries.map((entry, index) => <li key={index}>
    <span className="trace-step">{entry.step}</span>
    {entry.type === 'message' ? <div><b>Agent message</b><p>{entry.text}</p></div> : <div>
      <b>{entry.result?.ok ? <Check size={13} /> : <X size={13} />}{entry.name}</b>
      <code>{JSON.stringify(entry.arguments, null, 2)}</code>
      <span className="observation">Observation</span><pre>{JSON.stringify(entry.result, null, 2)}</pre>
    </div>}
  </li>)}</ol>;
}

const example = 'Calculate the total cost of 12 notebooks at 4.50 each plus 19% tax. Save the total in a note titled "Notebook budget", then read it back to verify it was saved.';
const statusLabel = status => ({completed:'Finished',limit:'Limit reached',error:'Failed',running:'Running / interrupted'}[status] || status);

export default function AgentWorkspace({ api, health }) {
  const [goal, setGoal] = useState(example);
  const [steps, setSteps] = useState(6);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [tasks, setTasks] = useState([]);
  const [selected, setSelected] = useState(null);
  const [notes, setNotes] = useState([]);

  async function refresh() {
    try {
      const [saved, memory] = await Promise.all([api('/tasks'), api('/notes')]);
      setTasks(saved); setNotes(memory); setSelected(current => current || saved[0] || null);
    } catch (err) { setError(err.message); }
  }
  useEffect(() => { refresh(); }, []);

  async function start(event) {
    event.preventDefault(); setBusy(true); setError(''); setSelected(null);
    try {
      const task = await api('/tasks', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal,max_steps:steps})});
      setSelected(task); setTasks(previous => [task,...previous].slice(0,30));
      const memory = await api('/notes'); setNotes(memory);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  return <div className="agent-workspace">
    {health && !health.llm_configured && <div className="key-notice"><Bot size={22}/><div><b>{health.provider === 'ollama' ? 'Start your free local model' : 'Connect the selected model'}</b><p>{health.runtime_message}</p>{health.provider === 'ollama' && <p>Open Ollama, run <code>ollama pull {health.model}</code> if needed, then refresh this page.</p>}<span>Real model decisions and tool calls. Local Ollama needs no API key.</span></div></div>}
    {health?.llm_configured && health.provider === 'ollama' && <div className="local-ready"><Check size={16}/><span><b>Free local AI is ready</b> · {health.model} via Ollama · No API key</span></div>}
    {error && <div className="error" role="alert">{error}</div>}
    <section className="card goal-card"><div className="section-heading"><div><h2><Bot size={18}/> Give the agent a goal</h2><p>It chooses tools, observes results, and continues until it answers or reaches a limit.</p></div><span className="pill">{health?.model || 'LLM powered'}</span></div>
      <form onSubmit={start}><label htmlFor="goal">Task</label><textarea id="goal" value={goal} onChange={event=>setGoal(event.target.value)} maxLength={4000} rows={4} required disabled={busy}/>
        <div className="goal-actions"><label htmlFor="steps">Maximum model steps <select id="steps" value={steps} disabled={busy} onChange={event=>setSteps(Number(event.target.value))}>{[2,4,6,8].map(value=><option key={value} value={value}>{value} steps</option>)}</select></label><button className="primary" type="submit" disabled={busy || !health?.llm_configured || !goal.trim()}>{busy?<LoaderCircle size={15} className="spin"/>:<Play size={14}/>} {busy?'Agent is working…':'Start autonomous task'}</button></div>
      </form><div className="agent-tools"><span><Calculator size={14}/> calculate</span><span><NotebookPen size={14}/> save_note</span><span><NotebookPen size={14}/> read_note</span><span><Search size={14}/> search_notes</span></div>
    </section>
    <div className="agent-columns"><section className="card task-result"><div className="section-heading"><div><h2>Task result & actions</h2><p>{selected?`${selected.steps || 0} model steps · ${selected.tool_calls || 0} tool calls · ${selected.duration_ms || 0} ms`:'Every tool call and observation is recorded here.'}</p></div>{selected && <span className={`result ${selected.status==='completed'?'pass':'fail'}`}>{statusLabel(selected.status)}</span>}</div>
      {busy ? <div className="empty"><LoaderCircle size={28} className="spin"/><h3>Working toward your goal</h3><p>Choosing tools and using their results. The trace appears when the task stops.</p></div> : selected ? <div className="task-body"><div className="task-goal"><b>Goal</b><p>{selected.goal}</p></div>{selected.answer && <div className="final-answer"><b>Final answer</b><p>{selected.answer}</p></div>}{selected.error && <div className="error">{selected.error}</div>}<h3>Action trace</h3><ActionTrace entries={selected.trace}/></div> : <div className="empty"><Bot size={30}/><h3>Ready for a goal</h3><p>Try the notebook budget task above.</p></div>}
    </section>
    <div className="agent-side"><section className="card memory"><div className="section-heading"><div><h2>Workspace memory</h2><p>Notes persist across tasks.</p></div><button className="text-button" aria-label="Refresh tasks and memory" onClick={refresh} disabled={busy}><RefreshCw size={15}/></button></div>{notes.length ? <div className="memory-list">{notes.map(note=><details key={note.title}><summary>{note.title}</summary><p>{note.content}</p></details>)}</div> : <p className="side-empty">No notes yet. Ask the agent to save something.</p>}</section>
      <section className="card task-history"><div className="section-heading"><div><h2>Recent tasks</h2><p>Last 30 tasks in this workspace.</p></div></div>{tasks.length ? tasks.map(task=><button className="task-link" key={task.id} disabled={busy} onClick={()=>setSelected(task)}><span>{task.goal}<small>{statusLabel(task.status)}</small></span><ChevronRight size={14}/></button>) : <p className="side-empty">Your tasks will appear here.</p>}</section></div></div>
    <p className="agent-footnote"><ArrowUpRight size={14}/> Up to 12 tool calls per task and a 60-second execution budget checked between requests. {health?.provider === 'ollama' ? 'Inference runs on your computer without API charges.' : 'The selected OpenAI provider uses your API account.'} The agent can calculate and manage its notes.</p>
  </div>;
}
