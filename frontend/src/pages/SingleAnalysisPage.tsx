import { FormEvent, useEffect, useMemo } from "react";
import { History, MessageSquare, RefreshCw, Trash2, TrendingUp } from "lucide-react";
import { useOutletContext } from "react-router-dom";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { PageTransition } from "../components/PageTransition";
import type { AppShellOutletContext } from "../layouts/AppShell";
import { useSingleAnalysisWorkspace } from "../lib/analysisWorkspace";

const MARKET_REGIME_LABELS: Record<string, string> = {
  risk_off: "Risk Off",
  risk_on: "Risk On",
  neutral: "Neutral",
  defensive: "Defensive",
  bullish: "Bullish",
  bearish: "Bearish",
};

export function SingleAnalysisPage() {
  const { setTopbarCenterContent } = useOutletContext<AppShellOutletContext>();
  const {
    form,
    setForm,
    sessions,
    currentSession,
    displayedMessages,
    result,
    draft,
    setDraft,
    stages,
    loadingSessions,
    deletingSessionId,
    sending,
    error,
    selectSession,
    resetView,
    sendStream,
    deleteSession,
  } = useSingleAnalysisWorkspace();

  const topbarStageFlow = useMemo(
    () => (
      <div className="stage-flow topbar-stage-flow" aria-label="Current stages">
        {stages.map((item, index) => (
          <div key={item.stage} className="stage-flow-segment">
            <div className={`stage-chip ${item.status}`}>
              <span>{item.label}</span>
            </div>
            {index < stages.length - 1 ? (
              <span className={`stage-arrow ${item.status}`} aria-hidden="true">
                {">>>>>"}
              </span>
            ) : null}
          </div>
        ))}
      </div>
    ),
    [stages],
  );

  useEffect(() => {
    setTopbarCenterContent(topbarStageFlow);
    return () => setTopbarCenterContent(null);
  }, [setTopbarCenterContent, topbarStageFlow]);

  const handleAnalyze = async (event: FormEvent) => {
    event.preventDefault();
    await sendStream(true);
  };

  const handleFollowup = async (event: FormEvent) => {
    event.preventDefault();
    await sendStream(false);
  };

  const decision = result?.decision_result;
  const marketRegimeLabel = decision
    ? MARKET_REGIME_LABELS[String(decision.market_regime).toLowerCase()] || decision.market_regime
    : "";

  return (
    <PageTransition>
      <div className="single-stock-layout">
        <aside className="single-stock-sidebar">
          <section className="panel">
            <div className="panel-head">
              <div className="single-stock-header">
                <h3>个股分析</h3>
                <span className="small-muted">
                  输入股票代码和持有信息，系统会按交易日自动生成样本并完成模型分析。
                </span>
              </div>
            </div>
            <form className="form-grid" onSubmit={handleAnalyze}>
              <label>
                <span>股票代码</span>
                <input
                  value={form.symbol}
                  onChange={(event) => setForm((prev) => ({ ...prev, symbol: event.target.value.replace(/\s/g, "") }))}
                  placeholder="000001"
                />
              </label>
              <label>
                <span>风险偏好</span>
                <select
                  value={form.risk_preference}
                  onChange={(event) => setForm((prev) => ({ ...prev, risk_preference: event.target.value }))}
                >
                  <option value="conservative">保守型</option>
                  <option value="balanced">均衡型</option>
                  <option value="aggressive">进取型</option>
                </select>
              </label>
              <label className="toggle-row">
                <span>当前是否持有</span>
                <input
                  type="checkbox"
                  checked={form.is_holding}
                  onChange={(event) => setForm((prev) => ({ ...prev, is_holding: event.target.checked }))}
                />
              </label>
              <label>
                <span>持有天数</span>
                <input
                  type="number"
                  min={0}
                  value={form.holding_days}
                  onChange={(event) => setForm((prev) => ({ ...prev, holding_days: Number(event.target.value || 0) }))}
                />
              </label>
              <button className="primary-button" disabled={sending}>
                {sending ? "分析中..." : "开始分析"}
                <TrendingUp size={18} />
              </button>
            </form>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div className="session-headline">
                <History size={16} />
                <h3>历史记录</h3>
              </div>
              <button className="ghost-button" type="button" onClick={resetView}>
                新会话
              </button>
            </div>
            <div className="session-list-wrap">
              <div className="session-list">
                {loadingSessions ? <div className="small-muted">正在加载历史...</div> : null}
                {!loadingSessions && !sessions.length ? <div className="small-muted">还没有分析记录。</div> : null}
                {sessions.map((item) => (
                  <div key={item.id} className={`session-item ${currentSession?.id === item.id ? "active" : ""}`}>
                    <button type="button" className="session-item-main" onClick={() => void selectSession(item)}>
                      <strong>{item.stock_name || item.symbol}</strong>
                      <span>{item.latest_trade_date || "无日期"}</span>
                      <small>{item.last_assistant_message || item.last_user_message || "点击查看聊天记录"}</small>
                    </button>
                    <button
                      type="button"
                      className="session-delete-button"
                      onClick={() => void deleteSession(item.id)}
                      disabled={deletingSessionId === item.id}
                      aria-label="删除历史记录"
                      title="删除历史记录"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </aside>

        <section className="single-stock-main">
          {result && decision ? (
            <>
              <section className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">Latest Snapshot</p>
                    <h3>
                      {result.stock.name} {result.stock.symbol}
                    </h3>
                  </div>
                  <div className="panel-head-meta">
                    <span className="small-muted">分析日期 {result.analysis_date}</span>
                    <span className="small-muted">样本日期 {result.effective_trade_date}</span>
                  </div>
                </div>

                <div className="hero-panel dark-hero hero-panel-expanded">
                  <div className="hero-summary">
                    <p className="eyebrow">Action</p>
                    <h3>{decision.decision.action_cn}</h3>
                    <p className="small-muted">
                      置信度 {(decision.decision.confidence * 100).toFixed(1)}% | 建议持有 {decision.decision.suggested_hold_days} 天
                    </p>
                  </div>

                  <div className="hero-market-overview">
                    <h4>行情概览</h4>
                    <div className="hero-market-grid">
                      <div>
                        <span>收盘价</span>
                        <strong>{(result.market_snapshot.close || 0).toFixed(2)}</strong>
                      </div>
                      <div>
                        <span>当日涨跌</span>
                        <strong>{(result.market_snapshot.pct_chg || 0).toFixed(2)}%</strong>
                      </div>
                      <div>
                        <span>5日均成交额</span>
                        <strong>{((result.market_snapshot.avg_amount_5 || 0) / 1e8).toFixed(2)} 亿</strong>
                      </div>
                      <div>
                        <span>20日表现</span>
                        <strong>{((result.market_snapshot.ret_20 || 0) * 100).toFixed(2)}%</strong>
                      </div>
                    </div>
                  </div>

                  <div className="hero-badge">{marketRegimeLabel}</div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div className="session-headline">
                    <MessageSquare size={16} />
                    <h3>对话</h3>
                  </div>
                  <button className="ghost-button" type="button" onClick={() => void sendStream(true)} disabled={sending}>
                    <RefreshCw size={16} />
                    刷新本次分析
                  </button>
                </div>
                <div className="chat-thread">
                  {!displayedMessages.length ? <div className="chat-empty">开始分析后，这里会保留每次提问和回答。</div> : null}
                  {displayedMessages.map((item) => (
                    <article key={`${item.id}-${item.role}`} className={`chat-bubble ${item.role}`}>
                      <span className="chat-role">{item.role === "user" ? "你" : "分析助手"}</span>
                      <div className="chat-content">
                        {item.role === "assistant" ? (
                          <MarkdownRenderer content={item.content} />
                        ) : (
                          <div className="plain-text">{item.content}</div>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
                <form className="chat-form" onSubmit={handleFollowup}>
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder="继续追问，比如：如果我已经持有 7 天，应该重点关注哪些风险？"
                    rows={4}
                  />
                  <button className="primary-button" disabled={sending || !result}>
                    {sending ? "发送中..." : "发送追问"}
                  </button>
                </form>
              </section>
            </>
          ) : (
            <section className="panel single-stock-empty">
              <h3>先选一只股票开始</h3>
              <p className="small-muted">系统会按交易日生成样本，依次完成提取数据、模型预测和 AI 分析。</p>
            </section>
          )}

          {error ? <div className="form-error">{error}</div> : null}
        </section>
      </div>
    </PageTransition>
  );
}
