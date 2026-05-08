import { FormEvent, useEffect, useMemo } from "react";
import { History, MessageSquare, RefreshCw, Sparkles, Trash2 } from "lucide-react";
import { useOutletContext } from "react-router-dom";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { PageTransition } from "../components/PageTransition";
import type { AppShellOutletContext } from "../layouts/AppShell";
import { useMarketScanWorkspace } from "../lib/analysisWorkspace";
import type { MarketCandidate } from "../lib/types";

const formatPercent = (value: number | null | undefined, digits = 1) => `${((value ?? 0) * 100).toFixed(digits)}%`;
const formatSignedPercent = (value: number | null | undefined, digits = 2) =>
  `${(value ?? 0) >= 0 ? "+" : ""}${((value ?? 0) * 100).toFixed(digits)}%`;
const formatAmountYi = (value: number | null | undefined, digits = 2) => `${((value ?? 0) / 1e8).toFixed(digits)}亿`;

function getHoldDays(item: MarketCandidate) {
  return item.recommended_hold_days ?? item.decision_result.decision.suggested_hold_days;
}

function getWinRate(item: MarketCandidate) {
  return item.predicted_win_rate ?? item.decision_result.decision.confidence;
}

function getSignalScore(item: MarketCandidate) {
  return item.signal_score ?? item.decision_result.scores.S_final;
}

const stageLabelMap: Record<string, string> = {
  candidate_select: "候选股票",
  extract_data: "提取数据",
  model_predict: "模型预测",
  llm_analysis: "AI分析",
};

export function MarketScanPage() {
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
  } = useMarketScanWorkspace();

  const topbarStageFlow = useMemo(
    () => (
      <div className="stage-flow topbar-stage-flow" aria-label="当前阶段">
        {stages.filter((item) => item.stage !== "decision_engine").map((item, index, visibleStages) => (
          <div key={item.stage} className="stage-flow-segment">
            <div className={`stage-chip ${item.status}`}>
              <div className="stage-chip-content">
                <span>{stageLabelMap[item.stage] || item.label}</span>
                {typeof item.current === "number" && typeof item.total === "number" && item.total > 0 ? (
                  <small>{`${item.current}/${item.total}${item.message ? ` | ${item.message}` : ""}`}</small>
                ) : item.message ? (
                  <small>{item.message}</small>
                ) : null}
              </div>
            </div>
            {index < visibleStages.length - 1 ? (
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

  const handleRecommend = async (event: FormEvent) => {
    event.preventDefault();
    await sendStream(true, false);
  };

  const handleFollowup = async (event: FormEvent) => {
    event.preventDefault();
    await sendStream(false);
  };

  return (
    <PageTransition>
      <div className="single-stock-layout">
        <aside className="single-stock-sidebar">
          <section className="panel">
            <div className="panel-head">
              <div className="single-stock-header">
                <h3>股票推荐</h3>
                <span className="small-muted">
                  系统会先提取可用样本，再批量进入预测模型，并按股票与持有天数组合排序返回推荐名单。
                </span>
              </div>
            </div>
            <form className="form-grid" onSubmit={handleRecommend}>
              <div className="mode-toggle" role="radiogroup" aria-label="推荐模式">
                <button
                  type="button"
                  className={form.scan_mode === "market" ? "mode-toggle-button active" : "mode-toggle-button"}
                  onClick={() => setForm((prev) => ({ ...prev, scan_mode: "market" }))}
                  disabled={sending}
                >
                  全市场推荐
                </button>
                <button
                  type="button"
                  className={form.scan_mode === "quick" ? "mode-toggle-button active" : "mode-toggle-button"}
                  onClick={() => setForm((prev) => ({ ...prev, scan_mode: "quick" }))}
                  disabled={sending}
                >
                  快速推荐
                </button>
              </div>
              <label>
                <span>推荐数量</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={form.top_n}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, top_n: Math.max(1, Math.min(100, Number(event.target.value || 1))) }))
                  }
                />
              </label>
              <div className="small-muted">
                {form.scan_mode === "quick"
                  ? "快速推荐会从当日可用样本里抽样预测，再为每只股票选择胜率最高的持有天数。"
                  : "全市场推荐会对全量候选股票批量预测，再保留每只股票的最优持有天数并输出前 K 名。"}
              </div>
              <button className="primary-button" disabled={sending}>
                {sending ? "推荐中..." : "一键推荐"}
                <Sparkles size={18} />
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
                {!loadingSessions && !sessions.length ? <div className="small-muted">还没有推荐记录。</div> : null}
                {sessions.map((item) => (
                  <div key={item.id} className={`session-item ${currentSession?.id === item.id ? "active" : ""}`}>
                    <button type="button" className="session-item-main" onClick={() => void selectSession(item)}>
                      <strong>{item.title || "股票推荐"}</strong>
                      <span>{item.latest_trade_date || "无日期"}</span>
                      <small>{item.last_assistant_message || item.last_user_message || "点击查看推荐对话"}</small>
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
          {result ? (
            <>
              {result.candidates.length ? (
                <section className="panel market-results-panel">
                  <div className="panel-head">
                    <div>
                      <p className="eyebrow">推荐名单</p>
                      <h3>本次返回 {result.selected_count} 只股票</h3>
                    </div>
                    <div className="panel-head-meta">
                      <span className="small-muted">{result.scan_mode === "quick" ? "快速推荐" : "全市场推荐"}</span>
                      <span className="small-muted">分析日期 {result.analysis_date || result.effective_trade_date}</span>
                      <span className="small-muted">样本日期 {result.effective_trade_date}</span>
                    </div>
                  </div>

                  <div className="market-result-summary">
                    <div>
                      <span>{result.scan_mode === "quick" ? "抽样股票" : "扫描股票"}</span>
                      <strong>{result.sample_size || result.total_candidates}</strong>
                    </div>
                    <div>
                      <span>全市场股票</span>
                      <strong>{result.market_total_candidates || result.total_candidates}</strong>
                    </div>
                    <div>
                      <span>候选池</span>
                      <strong>{result.pool_size}</strong>
                    </div>
                    <div>
                      <span>返回结果</span>
                      <strong>{result.selected_count}</strong>
                    </div>
                  </div>

                  <div className="market-result-grid">
                    {result.candidates.map((item) => (
                      <article key={`${item.symbol}-result`} className="market-result-card">
                        <div className="market-card-topline">
                          <span className="market-rank">#{item.rank}</span>
                          <span className="small-muted">{item.industry_sw || "未分类"}</span>
                        </div>
                        <h4>{item.name}</h4>
                        <div className="market-symbol">{item.symbol}</div>
                        <div className="market-card-metrics">
                          <div>
                            <span>胜率</span>
                            <strong>{formatPercent(getWinRate(item))}</strong>
                          </div>
                          <div>
                            <span>持有</span>
                            <strong>{getHoldDays(item)}天</strong>
                          </div>
                          <div>
                            <span>信号分</span>
                            <strong>{getSignalScore(item).toFixed(3)}</strong>
                          </div>
                          <div>
                            <span>预测收益</span>
                            <strong>{formatSignedPercent(item.ret_mu_pred)}</strong>
                          </div>
                          <div>
                            <span>当日涨跌</span>
                            <strong>{formatSignedPercent((item.market_snapshot?.pct_chg ?? item.pct_chg) / 100)}</strong>
                          </div>
                          <div>
                            <span>5日成交额</span>
                            <strong>{formatAmountYi(item.avg_amount_5)}</strong>
                          </div>
                        </div>
                        <div className="market-card-footer">
                          <span>{item.board || "未知板块"}</span>
                          {item.close ? <span>收盘价 {item.close.toFixed(2)}</span> : null}
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              ) : (
                <section className="panel single-stock-empty">
                  <h3>本次推荐没有筛出可展示股票</h3>
                  <p className="small-muted">
                    {result.scan_mode === "quick" ? "快速推荐" : "全市场推荐"}已完成，当前候选池为 {result.pool_size}，
                    返回结果为 {result.selected_count}。可以查看下方 AI 分析，或刷新后重试。
                  </p>
                </section>
              )}

              <section className="panel">
                <div className="panel-head">
                  <div className="session-headline">
                    <MessageSquare size={16} />
                    <h3>AI分析</h3>
                  </div>
                  <button className="ghost-button" type="button" onClick={() => void sendStream(true, true)} disabled={sending}>
                    <RefreshCw size={16} />
                    刷新本次推荐
                  </button>
                </div>
                <div className="chat-thread">
                  {!displayedMessages.length ? <div className="chat-empty">推荐完成后，这里会保留每次提问和回答。</div> : null}
                  {displayedMessages.map((item) => (
                    <article key={`${item.id}-${item.role}`} className={`chat-bubble ${item.role}`}>
                      <span className="chat-role">{item.role === "user" ? "你" : "推荐助手"}</span>
                      <div className="chat-content">
                        {item.role === "assistant" ? <MarkdownRenderer content={item.content} /> : <div className="plain-text">{item.content}</div>}
                      </div>
                    </article>
                  ))}
                </div>
                <form className="chat-form" onSubmit={handleFollowup}>
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder="继续追问，例如：为什么第一名更值得关注？如果只买两只，应该怎样比较它们？"
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
              <h3>先输入推荐数量开始</h3>
              <p className="small-muted">系统会自动提取样本、批量预测、按最优持有天数排序，并生成中文 AI 解读。</p>
            </section>
          )}

          {error ? <div className="form-error">{error}</div> : null}
        </section>
      </div>
    </PageTransition>
  );
}
