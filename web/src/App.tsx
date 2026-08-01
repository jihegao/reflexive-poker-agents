import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Seat = {
  seat: number;
  name: string;
  strategy: string;
  stackBb: number;
  active: boolean;
  allIn: boolean;
  cards: string[];
  isButton: boolean;
  isActor: boolean;
  controller: "human" | "rule_ai" | "llm_closed_loop";
  model: string;
  strategyProfile: Strategy;
  decisionHistory: Advice[];
  reflections: Reflection[];
};

type Strategy = {
  strategyId: string;
  basePersona: string;
  version: number;
  aggressionBias: number;
  riskMarginDelta: number;
  preferredRaiseScale: number;
  bluffFrequencyCap: number;
  memoryHands: number;
  notes: string[];
  reason: string;
  author: string;
};

type Advice = {
  action: string;
  raiseScale: number;
  confidence: number;
  summary: string;
  rationale: string;
  riskFlags: string[];
  provider: string;
  model: string;
  readOnly: boolean;
  seat?: number;
  handIndex?: number | null;
};

type Reflection = {
  handIndex: number;
  seat?: number;
  basePersona?: string;
  outcomeSummary: string;
  decisionReview: string;
  strategyAdjustment: string;
  whatWorked: string[];
  whatFailed: string[];
  provider?: string;
  model?: string;
};

type PlayerConfig = {
  strategy: string;
  controller: "human" | "rule_ai" | "llm_closed_loop";
  model: string;
};

type TableState = {
  tableId: string;
  version: number;
  phase: string;
  ended: boolean;
  owner: boolean;
  controller: "human" | "llm_closed_loop";
  adviceEnabled: boolean;
  pausedReason: string | null;
  canAct: boolean;
  legalActions: string[];
  raiseScales: number[];
  strategy: Strategy;
  strategyVersions: Strategy[];
  lastAdvice: Advice | null;
  heroDecisionHistory: Advice[];
  providerUsage: Record<string, number>;
  providerMode: "mock" | "live_aliyun";
  model: string;
  completedHandCount: number;
  liveCallBudget: { used: number; limit: number; warning: boolean; exhausted: boolean };
  hand: null | {
    handIndex: number;
    street: string;
    board: string[];
    potBb: number;
    toCallBb: number;
    seats: Seat[];
    actions: Array<Record<string, unknown>>;
    complete: boolean;
    showdown: boolean;
    winners: string[];
    rewards: Record<string, number>;
  };
};

type LiveEvent = { seq: number; type: string; payload: Record<string, unknown> };
type ModelCatalog = { provider: string; models: string[]; source: string; error: string | null };

const OPPONENTS = ["rock", "tag", "lag", "calling_station", "myopic"];
const DEFAULT_PLAYERS: PlayerConfig[] = [
  { strategy: "closed_loop_shaper", controller: "human", model: "deepseek-v4-flash" },
  ...["tag", "lag", "rock", "calling_station", "myopic"].map((strategy) => ({
    strategy,
    controller: "rule_ai" as const,
    model: "deepseek-v4-flash",
  })),
];
const LABELS: Record<string, string> = {
  rock: "Rock",
  tag: "TAG",
  lag: "LAG",
  calling_station: "Calling Station",
  myopic: "Myopic",
  fold: "弃牌",
  check_call: "过牌 / 跟注",
  raise: "加注",
  preflop: "翻牌前",
  flop: "翻牌",
  turn: "转牌",
  river: "河牌",
};

const API_ROOT = location.port === "5173" ? "http://127.0.0.1:8790" : "";

function cardParts(card: string) {
  const rawSuit = card.slice(-1);
  const suit = ({ c: "♣", d: "♦", h: "♥", s: "♠" } as Record<string, string>)[rawSuit] || rawSuit;
  return { rank: card.slice(0, -1), suit, red: suit === "♥" || suit === "♦" };
}

function Card({ value, hidden = false }: { value?: string; hidden?: boolean }) {
  if (hidden || !value) return <span className="card card-back"><i>R</i></span>;
  const card = cardParts(value);
  return (
    <span className={`card ${card.red ? "red" : ""}`}>
      <b>{card.rank}</b><i>{card.suit}</i>
    </span>
  );
}

function Switch({ checked, onChange, disabled = false, label }: { checked: boolean; onChange: () => void; disabled?: boolean; label: string }) {
  return (
    <button className={`switch ${checked ? "on" : ""}`} onClick={onChange} disabled={disabled} aria-pressed={checked} aria-label={label}>
      <span />
    </button>
  );
}

function modelOptions(catalog: ModelCatalog | null, selected: string) {
  return Array.from(new Set([selected, ...(catalog?.models || [])]));
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function Setup({ onStart, busy, catalog }: { onStart: (players: PlayerConfig[]) => void; busy: boolean; catalog: ModelCatalog | null }) {
  const [players, setPlayers] = useState<PlayerConfig[]>(DEFAULT_PLAYERS);
  const update = (seat: number, patch: Partial<PlayerConfig>) => setPlayers((old) => old.map((player, index) => index === seat ? { ...player, ...patch } : player));
  const models = (selected: string) => modelOptions(catalog, selected);
  return (
    <main className="setup-shell">
      <div className="setup-brand"><span className="brand-mark">R</span><span>REFLEXIVE TABLE</span></div>
      <section className="setup-card">
        <p className="eyebrow">LOCAL PLAYABLE LAB · 6-MAX NLH</p>
        <h1>在牌桌上观察一个策略<br />如何认识并改写自己。</h1>
        <p className="setup-lead">进入牌桌前为每个 Player 选择策略、控制方式和 opencode-go 模型。只有 LLM Agent 会调用所选模型。</p>
        <div className="setup-players">
          <div className="setup-players-head"><h2>Player 阵容与策略</h2><span>RULE / LLM · MODEL</span></div>
          {players.map((player, seat) => (
            <div className="setup-player-row" key={seat}>
              <div className="setup-player-name"><b>{seat === 0 ? "YOU · HERO" : `Seat ${seat}`}</b><span>{seat === 0 ? "主玩家" : "对手座位"}</span></div>
              {seat === 0 ? <div className="setup-fixed-strategy">closed-loop-shaper</div> : (
                <select aria-label={`Seat ${seat} 策略`} value={player.strategy} onChange={(event) => update(seat, { strategy: event.target.value })}>
                  {OPPONENTS.map((item) => <option key={item} value={item}>{LABELS[item]}</option>)}
                </select>
              )}
              <select aria-label={`${seat === 0 ? "Hero" : `Seat ${seat}`} 控制方式`} value={player.controller} onChange={(event) => update(seat, { controller: event.target.value as PlayerConfig["controller"] })}>
                {seat === 0 ? <><option value="human">Human</option><option value="llm_closed_loop">LLM Agent</option></> : <><option value="rule_ai">Rule AI</option><option value="llm_closed_loop">LLM Agent</option></>}
              </select>
              <select aria-label={`选择 ${seat === 0 ? "Hero" : `Seat ${seat}`} 模型`} value={player.model} disabled={player.controller !== "llm_closed_loop"} onChange={(event) => update(seat, { model: event.target.value })}>
                {models(player.model).map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
            </div>
          ))}
          <p className="setup-caption">LLM Agent 会使用服务器发现的 opencode-go 模型目录；全部为 Rule AI 时使用离线规则模式。</p>
        </div>
        <button className="start-button" disabled={busy} onClick={() => onStart(players)}>{busy ? "正在建桌…" : "进入牌桌"}<span>↗</span></button>
      </section>
      <p className="setup-foot">每手重置 100 BB · 无手数上限 · 仅手间结束</p>
    </main>
  );
}

function SeatView({ seat }: { seat: Seat }) {
  const position = `seat-${seat.seat}`;
  return (
    <div className={`seat ${position} ${seat.isActor ? "acting" : ""} ${!seat.active ? "folded" : ""} ${seat.seat === 0 ? "hero-seat" : ""}`}>
      <div className="seat-cards">
        {seat.cards.length ? seat.cards.map((card) => <Card key={card} value={card} />) : <><Card hidden /><Card hidden /></>}
      </div>
      <div className="seat-info">
        <div><b>{seat.seat === 0 ? "YOU · HERO" : LABELS[seat.strategy]}</b>{seat.isButton && <em>D</em>}</div>
        <span>{seat.stackBb.toFixed(1)} BB</span>
      </div>
    </div>
  );
}

function TableCenter({ state }: { state: TableState }) {
  const hand = state.hand!;
  return (
    <section className="table-stage">
      <div className="felt-shadow" />
      <div className="felt">
        <div className="felt-line" />
        <div className="pot"><span>POT</span><b>{hand.potBb.toFixed(1)} BB</b></div>
        <div className="board">
          {[0, 1, 2, 3, 4].map((index) => hand.board[index] ? <Card key={index} value={hand.board[index]} /> : <span key={index} className="card-slot" />)}
        </div>
        {hand.seats.map((seat) => <SeatView seat={seat} key={seat.seat} />)}
      </div>
      {hand.complete && (
        <div className="hand-result">
          <span>HAND {hand.handIndex + 1} COMPLETE</span>
          <b>{hand.winners.map((winner) => winner === "hero" ? "Hero" : winner.split("_").slice(2).join(" ")).join(" + ")} 获胜</b>
          <em className={(hand.rewards.hero || 0) >= 0 ? "win" : "loss"}>{(hand.rewards.hero || 0) >= 0 ? "+" : ""}{(hand.rewards.hero || 0).toFixed(1)} BB</em>
        </div>
      )}
    </section>
  );
}

function LeftRail({ state, events, busy, command, selectedSeat, onSelect }: { state: TableState; events: LiveEvent[]; busy: boolean; command: (path: string, body?: unknown) => void; selectedSeat: number | null; onSelect: (seat: number) => void }) {
  const hand = state.hand!;
  const selected = selectedSeat === null ? null : hand.seats.find((seat) => seat.seat === selectedSeat) || null;
  return (
    <aside className="rail left-rail">
      <div className="rail-title"><span>牌局状态</span><i className="live-dot" /> LIVE</div>
      <div className="hand-meta">
        <div><span>HAND</span><b>#{hand.handIndex + 1}</b></div>
        <div><span>STREET</span><b>{LABELS[hand.street]}</b></div>
        <div><span>TO CALL</span><b>{hand.toCallBb.toFixed(1)} BB</b></div>
      </div>
      <div className="section-head">其他 Agent <span>RULE / LLM</span></div>
      <div className="opponent-list">
        {hand.seats.slice(1).map((seat) => (
          <div className={`opponent-item ${selectedSeat === seat.seat ? "selected" : ""}`} key={seat.seat}>
            <button className="agent-select" onClick={() => onSelect(seat.seat)}>
              <i className={`type-dot type-${seat.strategy}`} />
              <span><b>{LABELS[seat.strategy]}</b><small>Seat {seat.seat} · {seat.controller === "llm_closed_loop" ? `LLM · ${seat.model} · v${seat.strategyProfile.version}` : "Rule AI"} · {seat.active ? "在局" : "已弃牌"}</small></span>
            </button>
            <strong>{seat.stackBb.toFixed(0)}</strong>
            <Switch
              label={`切换 Seat ${seat.seat} 控制器`}
              checked={seat.controller === "llm_closed_loop"}
              disabled={busy || !state.owner}
              onChange={() => command(`seats/${seat.seat}/controller`, { controller: seat.controller === "llm_closed_loop" ? "rule_ai" : "llm_closed_loop" })}
            />
          </div>
        ))}
      </div>
      {selected && <AgentInspector seat={selected} />}
      <div className="section-head event-heading">实时事件 <span>{state.version}</span></div>
      <div className="event-list">
        {events.slice(-6).reverse().map((event) => <div key={event.seq}><i /> <span>{event.type.replaceAll(".", " / ")}</span><time>#{event.seq}</time></div>)}
        {!events.length && hand.actions.slice(-5).reverse().map((action, index) => <div key={index}><i /><span>{String(action.actor)} · {LABELS[String(action.action)]}</span></div>)}
      </div>
    </aside>
  );
}

function AgentInspector({ seat }: { seat: Seat }) {
  return (
    <div className="agent-inspector">
      <div className="agent-inspector-head"><div><span>SELECTED AGENT</span><b>{LABELS[seat.strategy]}</b></div><em>{seat.controller === "llm_closed_loop" ? "LLM" : "RULE"}</em></div>
      {seat.controller === "llm_closed_loop" ? <>
        <div className="agent-model">{seat.model} · strategy v{seat.strategyProfile.version}</div>
        <div className="inspector-section"><span>本局思考历史</span>
          {seat.decisionHistory.length ? seat.decisionHistory.slice().reverse().map((advice, index) => (
            <div className="thought-item" key={`${advice.handIndex}-${index}`}><div><b>{LABELS[advice.action] || advice.action}</b><time>{advice.confidence ? `${Math.round(advice.confidence * 100)}%` : "LLM"}</time></div><p>{advice.rationale || advice.summary || "—"}</p></div>
          )) : <p className="empty-inspector">本手尚未产生 LLM 思考。</p>}
        </div>
        <div className="inspector-section"><span>反思信息</span>
          {seat.reflections.length ? seat.reflections.slice().reverse().map((reflection, index) => (
            <div className="reflection-item" key={`${reflection.handIndex}-${index}`}><b>手牌 #{reflection.handIndex + 1}</b><p>{reflection.outcomeSummary || "—"}</p><small>{reflection.strategyAdjustment || "保持当前策略"}</small></div>
          )) : <p className="empty-inspector">还没有已保存的反思。</p>}
        </div>
      </> : <p className="empty-inspector">Rule AI 不产生 LLM 思考或反思记录。</p>}
    </div>
  );
}

function AgentRail({ state, busy, command }: { state: TableState; busy: boolean; command: (path: string, body?: unknown) => void }) {
  const strategy = state.strategy;
  const previous = state.strategyVersions.at(-2);
  const delta = (key: keyof Strategy) => previous ? Number(strategy[key]) - Number(previous[key]) : 0;
  const thought = state.lastAdvice || state.heroDecisionHistory.at(-1);
  return (
    <aside className="rail agent-rail">
      <div className="agent-head">
        <div className="agent-orb"><span /></div>
        <div><p>HERO CONTROLLER</p><b>{state.controller === "human" ? "Human Player" : "LLM Agent"}</b><span>{state.controller === "human" ? "你拥有行动权" : "closed-loop-shaper"}</span></div>
        <Switch label="切换 Hero 控制器" checked={state.controller === "llm_closed_loop"} disabled={busy || !state.owner} onChange={() => command("hero/controller", { controller: state.controller === "human" ? "llm_closed_loop" : "human" })} />
      </div>
      <div className="model-strip"><span>{state.providerMode === "mock" ? "MOCK" : "LIVE"}</span><b>{state.model}</b><i>{state.providerMode === "mock" ? state.providerUsage.mock_calls || 0 : `${state.liveCallBudget.used}/${state.liveCallBudget.limit}`}</i></div>
      {state.pausedReason && <div className="failure-banner"><b>LLM 已暂停，控制器保持不变</b><span>{state.pausedReason}</span></div>}
      <div className="hero-thinking">
        <div className="hero-thinking-head"><div><span>当前形势下的 LLM 思考</span><b>{state.controller === "llm_closed_loop" ? "自动决策记录" : "只读建议"}</b></div><em>{thought ? `${Math.round((thought.confidence || 0) * 100)}%` : "—"}</em></div>
        {thought ? <><div className="advice-action"><span>建议行动</span><b>{LABELS[thought.action] || thought.action}{thought.action === "raise" ? ` · ${thought.raiseScale === 1.25 ? "ALL-IN" : thought.raiseScale * 100 + "% POT"}` : ""}</b></div><p>{thought.rationale || thought.summary || "—"}</p></> : <p className="empty-inspector">当前还没有 LLM 思考记录。切换 Hero 为 LLM Agent 后会显示。</p>}
      </div>
      <div className="strategy-head"><div><span>CURRENT STRATEGY</span><b>closed-loop-shaper</b></div><em>v{strategy.version}</em></div>
      <div className="strategy-metrics">
        <Metric label="激进偏置" value={strategy.aggressionBias} delta={delta("aggressionBias")} max={0.2} />
        <Metric label="风险边际" value={strategy.riskMarginDelta} delta={delta("riskMarginDelta")} max={0.1} />
        <Metric label="诈唬上限" value={strategy.bluffFrequencyCap} delta={delta("bluffFrequencyCap")} max={0.25} />
      </div>
      <div className="strategy-note"><span>策略注记</span><p>{strategy.notes?.[0] || "—"}</p></div>
      <div className="version-list">
        <div className="section-head">版本历史 <span>只读</span></div>
        {state.strategyVersions.slice().reverse().slice(0, 4).map((version) => (
          <div key={version.version} className="version-item"><i>v{version.version}</i><div><b>{version.author === "system" ? "初始策略" : "手后反思更新"}</b><span>{version.reason}</span></div></div>
        ))}
      </div>
    </aside>
  );
}

function Metric({ label, value, delta, max }: { label: string; value: number; delta: number; max: number }) {
  const width = Math.max(4, Math.min(100, 50 + value / max * 50));
  return <div className="metric"><div><span>{label}</span><b>{value >= 0 ? "+" : ""}{value.toFixed(2)}</b><em className={delta >= 0 ? "up" : "down"}>{delta ? `${delta > 0 ? "+" : ""}${delta.toFixed(2)}` : "—"}</em></div><span><i style={{ width: `${width}%` }} /></span></div>;
}

function ActionDock({ state, busy, command }: { state: TableState; busy: boolean; command: (path: string, body?: unknown) => void }) {
  const [raiseScale, setRaiseScale] = useState(0.5);
  const legal = new Set(state.legalActions);
  if (state.hand?.complete) return (
    <footer className="action-dock complete-dock">
      <div><span>本手已结算</span><b>下一手将重新以 100 BB 开始</b></div>
      <button className="ghost" disabled={busy || !state.owner} onClick={() => command("finish")}>结束牌桌</button>
      <button className="primary" disabled={busy || !state.owner} onClick={() => command("next-hand")}>开始下一手 <span>→</span></button>
    </footer>
  );
  return (
    <footer className="action-dock">
      <div className="turn-indicator"><i /><div><span>{state.canAct ? "YOUR DECISION" : "TABLE RUNNING"}</span><b>{state.canAct ? `需跟注 ${state.hand?.toCallBb.toFixed(1)} BB` : state.controller === "llm_closed_loop" ? "LLM 正在行动" : "等待对手"}</b></div></div>
      <button className="fold" disabled={busy || !state.canAct || !legal.has("fold")} onClick={() => command("actions", { action: "fold", raise_scale: 0.5 })}>弃牌 <kbd>F</kbd></button>
      <button disabled={busy || !state.canAct || !legal.has("check_call")} onClick={() => command("actions", { action: "check_call", raise_scale: 0.5 })}>{(state.hand?.toCallBb || 0) > 0 ? "跟注" : "过牌"} <kbd>C</kbd></button>
      <div className="raise-control">
        <div className="raise-scales">{state.raiseScales.map((scale) => <button key={scale} className={raiseScale === scale ? "selected" : ""} onClick={() => setRaiseScale(scale)}>{scale === 1.25 ? "ALL-IN" : `${scale * 100}%`}</button>)}</div>
        <button className="raise-button" disabled={busy || !state.canAct || !legal.has("raise")} onClick={() => command("actions", { action: "raise", raise_scale: raiseScale })}>加注 <span>{raiseScale === 1.25 ? "ALL-IN" : `${raiseScale * 100}% POT`}</span></button>
      </div>
    </footer>
  );
}

export default function App() {
  const [state, setState] = useState<TableState | null>(null);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [selectedSeat, setSelectedSeat] = useState<number | null>(null);
  const refreshTimer = useRef<number | null>(null);

  const load = useCallback(async (tableId: string) => {
    try {
      const value = await request<TableState>(`/api/tables/${tableId}`);
      setState(value);
    } catch {
      localStorage.removeItem("poker_demo_table");
    }
  }, []);

  useEffect(() => {
    const tableId = localStorage.getItem("poker_demo_table");
    if (tableId) void load(tableId);
  }, [load]);

  useEffect(() => {
    void request<ModelCatalog>("/api/models")
      .then(setCatalog)
      .catch((cause) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, []);

  useEffect(() => {
    if (!state?.tableId || state.ended) return;
    const socketOrigin = API_ROOT
      ? API_ROOT.replace("http:", "ws:").replace("https:", "wss:")
      : `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}`;
    const socket = new WebSocket(`${socketOrigin}/api/tables/${state.tableId}/events?after=${state.version}`);
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data) as LiveEvent;
      setEvents((old) => [...old.filter((item) => item.seq !== event.seq), event].slice(-30));
      if (refreshTimer.current) window.clearTimeout(refreshTimer.current);
      refreshTimer.current = window.setTimeout(() => void load(state.tableId), 60);
    };
    return () => socket.close();
  }, [state?.tableId, state?.ended, load]);

  const start = async (players: PlayerConfig[]) => {
    setBusy(true); setError(null);
    try {
      const providerMode = players.some((player) => player.controller === "llm_closed_loop") ? "live_aliyun" : "mock";
      const value = await request<TableState>("/api/tables", { method: "POST", body: JSON.stringify({ provider_mode: providerMode, opponents: players.slice(1).map((player) => player.strategy), seat_configs: players }) });
      localStorage.setItem("poker_demo_table", value.tableId);
      setSelectedSeat(null);
      setEvents([]);
      setState(value);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  };

  const command = useCallback(async (path: string, body?: unknown) => {
    if (!state) return;
    setBusy(true); setError(null);
    try {
      const value = await request<TableState>(`/api/tables/${state.tableId}/${path}`, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
      setState(value);
      if (value.ended) localStorage.removeItem("poker_demo_table");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }, [state]);

  const phaseLabel = useMemo(() => state ? ({ waiting_human: "等待 Hero", waiting_llm: "LLM 思考中", hand_complete: "本手完成", finished: "牌桌结束" }[state.phase] || "运行中") : "", [state]);

  if (!state || state.ended) return <><Setup onStart={start} busy={busy} catalog={catalog} />{error && <div className="toast">{error}</div>}</>;
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">R</span><div><b>REFLEXIVE TABLE</b><small>POKER AGENT LAB</small></div></div>
        <div className="table-id"><span>LOCAL TABLE</span><b>{state.tableId}</b><i>{phaseLabel}</i></div>
        <div className="top-actions"><span className={state.owner ? "owner-badge" : "spectator-badge"}>{state.owner ? "OWNER SESSION" : "READ-ONLY"}</span><button aria-label="结束并离开牌局" disabled={busy || !state.owner} onClick={() => void command("finish")}>×</button></div>
      </header>
      <div className="workspace">
        <LeftRail state={state} events={events} busy={busy} command={command} selectedSeat={selectedSeat} onSelect={setSelectedSeat} />
        <TableCenter state={state} />
        <AgentRail state={state} busy={busy} command={command} />
      </div>
      <ActionDock state={state} busy={busy} command={command} />
      {busy && <div className="busy-bar"><span /></div>}
      {error && <div className="toast">{error}<button onClick={() => setError(null)}>×</button></div>}
    </div>
  );
}
