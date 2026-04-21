import { useEffect, useState } from "react";
import { PageTransition } from "../components/PageTransition";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import type { DashboardSummary } from "../lib/types";

export function DashboardPage() {
  const { token } = useAuth();
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) return;
    api
      .get<DashboardSummary>("/dashboard/summary", token)
      .then(setSummary)
      .catch((err) => setError(err instanceof Error ? err.message : "加载失败"));
  }, [token]);

  return (
    <PageTransition>
      <section className="hero-panel">
        <div>
          <p className="eyebrow">智能决策</p>
          <h3>股票分析与推荐。</h3>
        </div>
        <div className="hero-badge">面向用户的股票分析界面</div>
      </section>

      {error ? <div className="form-error">{error}</div> : null}

      <div className="stats-grid">
        <article className="stat-card">
          <span>最新交易日</span>
          <strong>{summary?.latest_trade_date || "--"}</strong>
        </article>
        <article className="stat-card">
          <span>覆盖股票数</span>
          <strong>{summary?.security_count ?? "--"}</strong>
        </article>
        <article className="stat-card dark">
          <span>个股分析</span>
          <strong>已开放</strong>
        </article>
        <article className="stat-card">
          <span>股票推荐</span>
          <strong>已开放</strong>
        </article>
      </div>

      <div className="two-column">
        <section className="panel">
          <div className="panel-head">
            <h3>你可以做什么</h3>
          </div>
          <div className="module-list">
            <div className="module-item">
              <span>输入单只股票并填写持仓情况</span>
              <strong className="pill success">个股分析</strong>
            </div>
            <div className="module-item">
              <span>查看近期值得关注的股票名单</span>
              <strong className="pill success">股票推荐</strong>
            </div>
            <div className="module-item">
              <span>后续接入更多投资辅助功能</span>
              <strong className="pill success">持续更新</strong>
            </div>
          </div>
        </section>

        <section className="panel dark-panel">
          <div className="panel-head">
            <h3>使用建议</h3>
          </div>
          <div className="module-list">
            <div className="module-item">
              <span>先用个股分析查看单只股票的当前状态</span>
            </div>
            <div className="module-item">
              <span>再用股票推荐快速筛选近期值得跟踪的标的</span>
            </div>
            <div className="module-item">
              <span>更多决策能力将在后续版本继续完善</span>
            </div>
          </div>
        </section>
      </div>
    </PageTransition>
  );
}
