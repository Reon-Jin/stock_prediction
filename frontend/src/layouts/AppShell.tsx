import { ReactNode, useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronLeft,
  ChevronRight,
  Home,
  LayoutGrid,
  LogOut,
  ScanSearch,
  UserRoundSearch,
} from "lucide-react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

export type AppShellOutletContext = {
  setTopbarCenterContent: (content: ReactNode | null) => void;
};

const SIDEBAR_COLLAPSED_STORAGE_KEY = "app-shell-sidebar-collapsed";

function readStoredCollapsedState() {
  if (typeof window === "undefined") {
    return false;
  }

  return window.localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === "true";
}

function writeStoredCollapsedState(value: boolean) {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, String(value));
}

const navItems = [
  { to: "/app/overview", label: "首页", icon: Home },
  { to: "/app/single-analysis", label: "个股分析", icon: UserRoundSearch },
  { to: "/app/market-scan", label: "股票推荐", icon: ScanSearch },
  { to: "/app/placeholders", label: "更多功能", icon: LayoutGrid },
];

export function AppShell() {
  const { pathname } = useLocation();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(readStoredCollapsedState);
  const [showSidebarText, setShowSidebarText] = useState(() => !readStoredCollapsedState());
  const [isClosing, setIsClosing] = useState(false);
  const [topbarCenterContent, setTopbarCenterContent] = useState<ReactNode | null>(null);
  const transitionTimerRef = useRef<number | null>(null);

  const clearTransitionTimer = () => {
    if (transitionTimerRef.current !== null) {
      window.clearTimeout(transitionTimerRef.current);
      transitionTimerRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      clearTransitionTimer();
    };
  }, []);

  const handleSidebarToggle = () => {
    clearTransitionTimer();

    if (collapsed) {
      writeStoredCollapsedState(false);
      setCollapsed(false);
      setIsClosing(false);
      transitionTimerRef.current = window.setTimeout(() => {
        setShowSidebarText(true);
        transitionTimerRef.current = null;
      }, 220);
      return;
    }

    if (isClosing) {
      writeStoredCollapsedState(false);
      setIsClosing(false);
      setShowSidebarText(true);
      return;
    }

    writeStoredCollapsedState(true);
    setIsClosing(true);
    setShowSidebarText(false);
    transitionTimerRef.current = window.setTimeout(() => {
      setCollapsed(true);
      setIsClosing(false);
      transitionTimerRef.current = null;
    }, 150);
  };

  const currentPageTitle =
    navItems.find((item) => pathname.startsWith(item.to))?.label || "智能选股助手";

  return (
    <div className={`shell ${collapsed ? "shell-collapsed" : ""}`}>
      <aside className={`sidebar ${collapsed ? "collapsed" : ""} ${showSidebarText ? "text-ready" : ""} ${isClosing ? "closing" : ""}`}>
        <div className="sidebar-top">
          <div className="sidebar-toggle-row">
            <button
              type="button"
              className="sidebar-toggle"
              onClick={handleSidebarToggle}
              aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"}
              title={collapsed ? "展开侧边栏" : "收起侧边栏"}
            >
              {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </button>
          </div>

          <div className="brand-card">
            <div className="brand-orb" />
            <div
              className={`brand-copy sidebar-copy ${showSidebarText ? "visible" : ""}`}
              aria-hidden={!showSidebarText}
            >
              <p className="eyebrow">A股智能助手</p>
              <h1>智能选股助手</h1>
            </div>
          </div>

          <nav className="nav-list">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
                  title={collapsed ? item.label : undefined}
                >
                  <span className="nav-icon">
                    <Icon size={18} />
                  </span>
                  <span
                    className={`nav-label sidebar-copy ${showSidebarText ? "visible" : ""}`}
                    aria-hidden={!showSidebarText}
                  >
                    {item.label}
                  </span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        <div className="sidebar-footer">
          <div className="user-chip" title={collapsed ? user?.display_name || user?.username : undefined}>
            <Bot size={16} />
            <span className={`sidebar-copy ${showSidebarText ? "visible" : ""}`} aria-hidden={!showSidebarText}>
              {user?.display_name || user?.username}
            </span>
          </div>

          <button
            className="ghost-button full-width logout-button"
            onClick={() => {
              logout();
              navigate("/login");
            }}
            title={collapsed ? "退出登录" : undefined}
          >
            <LogOut size={16} />
            <span className={`sidebar-copy ${showSidebarText ? "visible" : ""}`} aria-hidden={!showSidebarText}>
              退出登录
            </span>
          </button>
        </div>
      </aside>

      <div className="shell-main">
        <header className="topbar">
          <div className="topbar-title">
            <p className="eyebrow">当前页面</p>
            <h2>{currentPageTitle}</h2>
          </div>
          <div className="topbar-center">{topbarCenterContent}</div>
          <div className="topbar-meta">
            <span className="status-dot" />
            <span>服务已连接</span>
          </div>
        </header>

        <main className="content">
          <Outlet context={{ setTopbarCenterContent }} />
        </main>
      </div>
    </div>
  );
}
