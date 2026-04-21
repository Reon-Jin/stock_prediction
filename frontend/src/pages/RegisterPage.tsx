import { ArrowRight } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageTransition } from "../components/PageTransition";
import { useAuth } from "../lib/auth";

export function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    display_name: "",
  });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!form.username.trim()) {
      setError("请输入用户名");
      return;
    }
    if (!form.email.trim()) {
      setError("请输入邮箱");
      return;
    }
    if (!form.password) {
      setError("请输入密码");
      return;
    }
    if (form.password.length < 6) {
      setError("密码至少需要 6 位");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await register(form);
      navigate("/app/overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "注册失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageTransition>
      <div className="auth-page register">
        <section className="auth-card">
          <p className="eyebrow">注册</p>
          <h2>创建账号</h2>
          <form onSubmit={handleSubmit} className="form-grid">
            <label>
              <span>用户名</span>
              <input required value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} />
            </label>
            <label>
              <span>邮箱</span>
              <input required value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
            </label>
            <label>
              <span>显示名称</span>
              <input value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} />
            </label>
            <label>
              <span>密码</span>
              <input required minLength={6} type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />
            </label>
            {error ? <div className="form-error">{error}</div> : null}
            <button className="primary-button" disabled={submitting}>
              {submitting ? "提交中..." : "注册"}
              <ArrowRight size={18} />
            </button>
          </form>
          <p className="muted-line">
            已有账号？<Link to="/login">返回登录</Link>
          </p>
        </section>

        <section className="auth-side-panel">
          <div className="feature-stack">
            <div className="info-card dark">
              <h3>个股分析</h3>
              <p>输入股票代码和持仓信息，快速查看分析结果。</p>
            </div>
            <div className="info-card">
              <h3>股票推荐</h3>
              <p>查看近期值得关注的股票名单，帮助你更高效地筛选标的。</p>
            </div>
          </div>
        </section>
      </div>
    </PageTransition>
  );
}
