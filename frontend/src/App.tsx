import { AnimatePresence } from "framer-motion";
import { LoaderCircle } from "lucide-react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AppShell } from "./layouts/AppShell";
import { useAuth } from "./lib/auth";
import { DashboardPage } from "./pages/DashboardPage";
import { LoginPage } from "./pages/LoginPage";
import { MarketScanPage } from "./pages/MarketScanPage";
import { PlaceholdersPage } from "./pages/PlaceholdersPage";
import { RegisterPage } from "./pages/RegisterPage";
import { SingleAnalysisPage } from "./pages/SingleAnalysisPage";

function ProtectedRoutes() {
  const { user } = useAuth();
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  return <AppShell />;
}

export default function App() {
  const { loading, user } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="loading-screen">
        <LoaderCircle className="spin" size={30} />
        <span>正在连接服务...</span>
      </div>
    );
  }

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/login" element={user ? <Navigate to="/app/overview" replace /> : <LoginPage />} />
        <Route path="/register" element={user ? <Navigate to="/app/overview" replace /> : <RegisterPage />} />
        <Route path="/app" element={<ProtectedRoutes />}>
          <Route path="overview" element={<DashboardPage />} />
          <Route path="single-analysis" element={<SingleAnalysisPage />} />
          <Route path="market-scan" element={<MarketScanPage />} />
          <Route path="placeholders" element={<PlaceholdersPage />} />
        </Route>
        <Route path="*" element={<Navigate to={user ? "/app/overview" : "/login"} replace />} />
      </Routes>
    </AnimatePresence>
  );
}
