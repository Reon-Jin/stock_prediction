import { ArrowRight } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageTransition } from "../components/PageTransition";
import { useAuth } from "../lib/auth";

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [account, setAccount] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!account.trim()) {
      setError("请输入用户名或邮箱");
      return;
    }
    if (!password) {
      setError("请输入密码");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await login(account, password);
      navigate("/app/overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageTransition>
      <div className="auth-page">
        <section className="auth-hero">
          <p className="eyebrow">股票洞察</p>
          <h1>股票分析助手</h1>
          <p className="hero-copy">登录后即可使用个股分析、股票推荐和后续扩展功能。</p>
        </section>

        <section className="auth-card">
          <p className="eyebrow">登录</p>
          <h2>欢迎使用</h2>
          <form onSubmit={handleSubmit} className="form-grid">
            <label>
              <span>用户名或邮箱</span>
              <input required value={account} onChange={(event) => setAccount(event.target.value)} placeholder="请输入账号" />
            </label>
            <label>
              <span>密码</span>
              <input required type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="请输入密码" />
            </label>
            {error ? <div className="form-error">{error}</div> : null}
            <button className="primary-button" disabled={submitting}>
              {submitting ? "登录中..." : "登录"}
              <ArrowRight size={18} />
            </button>
          </form>
          <p className="muted-line">
            还没有账号？<Link to="/register">立即注册</Link>
          </p>
        </section>
      </div>
    </PageTransition>
  );
}
