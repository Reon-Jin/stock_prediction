import {
  PropsWithChildren,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { api } from "./api";
import { useAuth } from "./auth";
import type {
  AnalysisMessage,
  AnalysisSessionDetail,
  AnalysisSessionSummary,
  AnalysisStage,
  AnalysisStageKey,
  MarketScanResult,
  SingleAnalysisResult,
} from "./types";

const BASE_STAGES: AnalysisStage[] = [
  { stage: "candidate_select", label: "候选股票", status: "idle" },
  { stage: "model_predict", label: "模型预测", status: "idle" },
  { stage: "llm_analysis", label: "AI分析", status: "idle" },
];

const SINGLE_ANALYSIS_STAGES: AnalysisStage[] = [
  { stage: "extract_data", label: "提取数据", status: "idle" },
  { stage: "model_predict", label: "模型预测", status: "idle" },
  { stage: "decision_engine", label: "决策引擎", status: "idle" },
  { stage: "llm_analysis", label: "AI分析", status: "idle" },
];

function normalizeClientSymbol(value: string | null | undefined) {
  return String(value || "").trim().replace(/\s/g, "").slice(0, 6);
}

function buildStageState(
  activeStage?: AnalysisStageKey,
  payload?: Partial<AnalysisStage> & { stage?: AnalysisStageKey },
  previous: AnalysisStage[] = BASE_STAGES,
  stageTemplate: AnalysisStage[] = BASE_STAGES,
): AnalysisStage[] {
  const visibleActiveStage =
    activeStage === "decision_engine" && !stageTemplate.some((item) => item.stage === "decision_engine")
      ? "model_predict"
      : activeStage;
  const activeIndex = visibleActiveStage ? stageTemplate.findIndex((item) => item.stage === visibleActiveStage) : -1;
  return stageTemplate.map((item, index) => {
    const prev = previous.find((entry) => entry.stage === item.stage) || item;
    const patch = payload?.stage === item.stage ? payload : undefined;
    if (activeIndex < 0) {
      return { ...prev, status: "idle" };
    }
    if (index < activeIndex) {
      return { ...prev, status: "done" };
    }
    if (index === activeIndex) {
      return { ...prev, ...(patch || {}), status: "running" };
    }
    return { ...prev, status: "idle" };
  });
}

type SingleAnalysisForm = {
  symbol: string;
  is_holding: boolean;
  holding_days: number;
  risk_preference: string;
};

type MarketScanForm = {
  top_n: number;
  scan_mode: "market" | "quick" | "direct";
};

type SingleAnalysisWorkspaceValue = {
  form: SingleAnalysisForm;
  setForm: (updater: SingleAnalysisForm | ((prev: SingleAnalysisForm) => SingleAnalysisForm)) => void;
  sessions: AnalysisSessionSummary[];
  currentSession: AnalysisSessionSummary | null;
  messages: AnalysisMessage[];
  displayedMessages: AnalysisMessage[];
  result: SingleAnalysisResult | null;
  draft: string;
  setDraft: (value: string) => void;
  stages: AnalysisStage[];
  loadingSessions: boolean;
  deletingSessionId: number | null;
  sending: boolean;
  error: string;
  selectSession: (session: AnalysisSessionSummary | null) => Promise<void>;
  resetView: () => void;
  loadSessions: (preferredSessionId?: number) => Promise<void>;
  sendStream: (refreshAnalysis: boolean) => Promise<void>;
  deleteSession: (sessionId: number) => Promise<void>;
};

type MarketScanWorkspaceValue = {
  form: MarketScanForm;
  setForm: (updater: MarketScanForm | ((prev: MarketScanForm) => MarketScanForm)) => void;
  sessions: AnalysisSessionSummary[];
  currentSession: AnalysisSessionSummary | null;
  messages: AnalysisMessage[];
  displayedMessages: AnalysisMessage[];
  result: MarketScanResult | null;
  draft: string;
  setDraft: (value: string) => void;
  stages: AnalysisStage[];
  loadingSessions: boolean;
  deletingSessionId: number | null;
  sending: boolean;
  error: string;
  selectSession: (session: AnalysisSessionSummary | null) => Promise<void>;
  resetView: () => void;
  loadSessions: (preferredSessionId?: number) => Promise<void>;
  sendStream: (refreshAnalysis: boolean) => Promise<void>;
  deleteSession: (sessionId: number) => Promise<void>;
};

const SingleAnalysisWorkspaceContext = createContext<SingleAnalysisWorkspaceValue | null>(null);
const MarketScanWorkspaceContext = createContext<MarketScanWorkspaceValue | null>(null);

export function AnalysisWorkspaceProvider({ children }: PropsWithChildren) {
  const { token } = useAuth();

  const [singleForm, setSingleFormState] = useState<SingleAnalysisForm>({
    symbol: "000001",
    is_holding: false,
    holding_days: 0,
    risk_preference: "balanced",
  });
  const [singleSessions, setSingleSessions] = useState<AnalysisSessionSummary[]>([]);
  const [singleCurrentSession, setSingleCurrentSession] = useState<AnalysisSessionSummary | null>(null);
  const [singleMessages, setSingleMessages] = useState<AnalysisMessage[]>([]);
  const [singleResult, setSingleResult] = useState<SingleAnalysisResult | null>(null);
  const [singleDraft, setSingleDraft] = useState("");
  const [singleStreamingAssistant, setSingleStreamingAssistant] = useState("");
  const [singleStages, setSingleStages] = useState<AnalysisStage[]>(SINGLE_ANALYSIS_STAGES);
  const [singleLoadingSessions, setSingleLoadingSessions] = useState(false);
  const [singleDeletingSessionId, setSingleDeletingSessionId] = useState<number | null>(null);
  const [singleSending, setSingleSending] = useState(false);
  const [singleError, setSingleError] = useState("");

  const [marketForm, setMarketFormState] = useState<MarketScanForm>({ top_n: 12, scan_mode: "market" });
  const [marketSessions, setMarketSessions] = useState<AnalysisSessionSummary[]>([]);
  const [marketCurrentSession, setMarketCurrentSession] = useState<AnalysisSessionSummary | null>(null);
  const [marketMessages, setMarketMessages] = useState<AnalysisMessage[]>([]);
  const [marketResult, setMarketResult] = useState<MarketScanResult | null>(null);
  const [marketDraft, setMarketDraft] = useState("");
  const [marketStreamingAssistant, setMarketStreamingAssistant] = useState("");
  const [marketStages, setMarketStages] = useState<AnalysisStage[]>(BASE_STAGES);
  const [marketLoadingSessions, setMarketLoadingSessions] = useState(false);
  const [marketDeletingSessionId, setMarketDeletingSessionId] = useState<number | null>(null);
  const [marketSending, setMarketSending] = useState(false);
  const [marketError, setMarketError] = useState("");

  const setSingleForm = useCallback((updater: SingleAnalysisForm | ((prev: SingleAnalysisForm) => SingleAnalysisForm)) => {
    setSingleFormState((prev) => (typeof updater === "function" ? updater(prev) : updater));
  }, []);

  const setMarketForm = useCallback((updater: MarketScanForm | ((prev: MarketScanForm) => MarketScanForm)) => {
    setMarketFormState((prev) => (typeof updater === "function" ? updater(prev) : updater));
  }, []);

  const resetSingleView = useCallback(() => {
    setSingleCurrentSession(null);
    setSingleMessages([]);
    setSingleResult(null);
    setSingleStreamingAssistant("");
    setSingleDraft("");
    setSingleStages(SINGLE_ANALYSIS_STAGES);
    setSingleError("");
  }, []);

  const resetMarketView = useCallback(() => {
    setMarketCurrentSession(null);
    setMarketMessages([]);
    setMarketResult(null);
    setMarketStreamingAssistant("");
    setMarketDraft("");
    setMarketStages(BASE_STAGES);
    setMarketError("");
  }, []);

  const loadSingleSessions = useCallback(
    async (preferredSessionId?: number) => {
      if (!token) return;
      setSingleLoadingSessions(true);
      try {
        const data = await api.get<AnalysisSessionSummary[]>("/analysis/single/sessions", token);
        setSingleSessions(data);
        if (preferredSessionId) {
          const found = data.find((item) => item.id === preferredSessionId) || null;
          if (found) {
            setSingleCurrentSession(found);
          }
        } else if (!singleCurrentSession && data.length) {
          setSingleCurrentSession(data[0]);
        }
      } catch (err) {
        setSingleError(err instanceof Error ? err.message : "鍔犺浇鍘嗗彶澶辫触");
      } finally {
        setSingleLoadingSessions(false);
      }
    },
    [singleCurrentSession, token],
  );

  const loadMarketSessions = useCallback(
    async (preferredSessionId?: number) => {
      if (!token) return;
      setMarketLoadingSessions(true);
      try {
        const data = await api.get<AnalysisSessionSummary[]>("/analysis/market-scan/sessions", token);
        setMarketSessions(data);
        if (preferredSessionId) {
          const found = data.find((item) => item.id === preferredSessionId) || null;
          if (found) {
            setMarketCurrentSession(found);
          }
        } else if (!marketCurrentSession && data.length) {
          setMarketCurrentSession(data[0]);
        }
      } catch (err) {
        setMarketError(err instanceof Error ? err.message : "鍔犺浇鍘嗗彶澶辫触");
      } finally {
        setMarketLoadingSessions(false);
      }
    },
    [marketCurrentSession, token],
  );

  const loadSingleSessionDetail = useCallback(
    async (sessionId: number) => {
      if (!token) return;
      try {
        const detail = await api.get<AnalysisSessionDetail>(`/analysis/single/sessions/${sessionId}`, token);
        setSingleCurrentSession(detail);
        setSingleMessages(detail.messages || []);
        setSingleResult((detail.latest_analysis as SingleAnalysisResult | null) || null);
        setSingleStreamingAssistant("");
        setSingleFormState({
          symbol: detail.symbol?.slice(0, 6) || "000001",
          is_holding: detail.is_holding,
          holding_days: detail.holding_days,
          risk_preference: detail.risk_preference,
        });
      } catch (err) {
        setSingleError(err instanceof Error ? err.message : "鍔犺浇浼氳瘽澶辫触");
      }
    },
    [token],
  );

  const loadMarketSessionDetail = useCallback(
    async (sessionId: number) => {
      if (!token) return;
      try {
        const detail = await api.get<AnalysisSessionDetail>(`/analysis/market-scan/sessions/${sessionId}`, token);
        setMarketCurrentSession(detail);
        setMarketMessages(detail.messages || []);
        setMarketResult((detail.latest_analysis as MarketScanResult | null) || null);
        setMarketStreamingAssistant("");
        const marketResult = (detail.latest_analysis as MarketScanResult | null) || null;
        const detailTopN = Number(marketResult?.top_n || detail.holding_days || 12);
        const detailMode = marketResult?.scan_mode === "quick" ? "quick" : (marketResult?.scan_mode === "direct" ? "direct" : "market");
        setMarketFormState({ top_n: detailTopN, scan_mode: detailMode });
      } catch (err) {
        setMarketError(err instanceof Error ? err.message : "鍔犺浇浼氳瘽澶辫触");
      }
    },
    [token],
  );

  const selectSingleSession = useCallback(
    async (session: AnalysisSessionSummary | null) => {
      setSingleCurrentSession(session);
      if (!session?.id) {
        return;
      }
      await loadSingleSessionDetail(session.id);
    },
    [loadSingleSessionDetail],
  );

  const selectMarketSession = useCallback(
    async (session: AnalysisSessionSummary | null) => {
      setMarketCurrentSession(session);
      if (!session?.id) {
        return;
      }
      await loadMarketSessionDetail(session.id);
    },
    [loadMarketSessionDetail],
  );

  const sendSingleStream = useCallback(
    async (refreshAnalysis: boolean) => {
      if (!token) return;
      const requestSymbol = normalizeClientSymbol(singleForm.symbol);
      const currentSessionSymbol = normalizeClientSymbol(singleCurrentSession?.symbol);
      const reuseCurrentSession = Boolean(singleCurrentSession?.id) && requestSymbol !== "" && requestSymbol === currentSessionSymbol;
      setSingleSending(true);
      setSingleError("");
      setSingleStreamingAssistant("");
      setSingleStages(
        buildStageState(refreshAnalysis ? "extract_data" : "llm_analysis", undefined, SINGLE_ANALYSIS_STAGES, SINGLE_ANALYSIS_STAGES),
      );
      try {
        await api.stream(
          "/analysis/single/stream",
          {
            session_id: reuseCurrentSession ? (singleCurrentSession?.id ?? null) : null,
            symbol: requestSymbol,
            is_holding: singleForm.is_holding,
            holding_days: singleForm.holding_days,
            risk_preference: singleForm.risk_preference,
            message: singleDraft,
            refresh_analysis: refreshAnalysis,
          },
          token,
          {
            onEvent: (event, payload) => {
              if (event === "session") {
                const nextId = Number(payload.session_id);
                setSingleCurrentSession((prev) => (prev && prev.id === nextId ? prev : ({ ...(prev || {}), id: nextId } as AnalysisSessionSummary)));
              }
              if (event === "stage") {
                setSingleStages((prev) =>
                  buildStageState(payload.stage as AnalysisStageKey, payload as Partial<AnalysisStage>, prev, SINGLE_ANALYSIS_STAGES),
                );
              }
              if (event === "analysis_result") {
                setSingleResult(payload as SingleAnalysisResult);
              }
              if (event === "message") {
                setSingleMessages((prev) => {
                  const exists = prev.some((item) => item.id === payload.id);
                  return exists ? prev : [...prev, payload as AnalysisMessage];
                });
              }
              if (event === "delta") {
                setSingleStreamingAssistant((prev) => prev + String(payload.content || ""));
              }
              if (event === "assistant_message") {
                setSingleMessages((prev) => {
                  const filtered = prev.filter((item) => item.id !== -1);
                  const exists = filtered.some((item) => item.id === payload.id);
                  return exists ? filtered : [...filtered, payload as AnalysisMessage];
                });
                setSingleStreamingAssistant("");
              }
              if (event === "done") {
                setSingleStages(SINGLE_ANALYSIS_STAGES.map((item) => ({ ...item, status: "done" })));
                setSingleSending(false);
                setSingleDraft("");
                void loadSingleSessions(Number(payload.session_id));
                if (payload.session_id) {
                  void loadSingleSessionDetail(Number(payload.session_id));
                }
              }
              if (event === "error") {
                setSingleError(String(payload.detail || "分析失败"));
                setSingleSending(false);
              }
            },
          },
        );
      } catch (err) {
        setSingleError(err instanceof Error ? err.message : "分析失败");
        setSingleSending(false);
      }
    },
    [loadSingleSessionDetail, loadSingleSessions, singleCurrentSession?.id, singleDraft, singleForm, token],
  );

  const sendMarketStream = useCallback(
    async (refreshAnalysis: boolean) => {
      if (!token) return;
      setMarketSending(true);
      setMarketError("");
      setMarketStreamingAssistant("");
      setMarketStages((prev) => buildStageState(refreshAnalysis ? "candidate_select" : "llm_analysis", undefined, prev));
      try {
        await api.stream(
          "/analysis/market-scan/stream",
          {
            session_id: marketCurrentSession?.id ?? null,
            top_n: marketForm.top_n,
            scan_mode: marketForm.scan_mode,
            message: marketDraft,
            refresh_analysis: refreshAnalysis,
          },
          token,
          {
            onEvent: (event, payload) => {
              if (event === "session") {
                const nextId = Number(payload.session_id);
                setMarketCurrentSession((prev) => (prev && prev.id === nextId ? prev : ({ ...(prev || {}), id: nextId } as AnalysisSessionSummary)));
              }
              if (event === "stage") {
                setMarketStages((prev) =>
                  buildStageState(payload.stage as AnalysisStageKey, payload as Partial<AnalysisStage>, prev),
                );
              }
              if (event === "analysis_result") {
                setMarketResult(payload as MarketScanResult);
              }
              if (event === "message") {
                setMarketMessages((prev) => {
                  const exists = prev.some((item) => item.id === payload.id);
                  return exists ? prev : [...prev, payload as AnalysisMessage];
                });
              }
              if (event === "delta") {
                setMarketStreamingAssistant((prev) => prev + String(payload.content || ""));
              }
              if (event === "assistant_message") {
                setMarketMessages((prev) => {
                  const filtered = prev.filter((item) => item.id !== -1);
                  const exists = filtered.some((item) => item.id === payload.id);
                  return exists ? filtered : [...filtered, payload as AnalysisMessage];
                });
                setMarketStreamingAssistant("");
              }
              if (event === "done") {
                setMarketStages(BASE_STAGES.map((item) => ({ ...item, status: "done" })));
                setMarketSending(false);
                setMarketDraft("");
                void loadMarketSessions(Number(payload.session_id));
                if (payload.session_id) {
                  void loadMarketSessionDetail(Number(payload.session_id));
                }
              }
              if (event === "error") {
                setMarketError(String(payload.detail || "推荐失败"));
                setMarketSending(false);
              }
            },
          },
        );
      } catch (err) {
        setMarketError(err instanceof Error ? err.message : "推荐失败");
        setMarketSending(false);
      }
    },
    [loadMarketSessionDetail, loadMarketSessions, marketCurrentSession?.id, marketDraft, marketForm, token],
  );

  const deleteSingleSession = useCallback(
    async (sessionId: number) => {
      if (!token) return;
      setSingleDeletingSessionId(sessionId);
      setSingleError("");
      try {
        await api.delete(`/analysis/single/sessions/${sessionId}`, token);
        setSingleSessions((prev) => prev.filter((item) => item.id !== sessionId));
        if (singleCurrentSession?.id === sessionId) {
          resetSingleView();
        }
      } catch (err) {
        setSingleError(err instanceof Error ? err.message : "删除历史记录失败");
      } finally {
        setSingleDeletingSessionId(null);
      }
    },
    [resetSingleView, singleCurrentSession?.id, token],
  );

  const deleteMarketSession = useCallback(
    async (sessionId: number) => {
      if (!token) return;
      setMarketDeletingSessionId(sessionId);
      setMarketError("");
      try {
        await api.delete(`/analysis/market-scan/sessions/${sessionId}`, token);
        setMarketSessions((prev) => prev.filter((item) => item.id !== sessionId));
        if (marketCurrentSession?.id === sessionId) {
          resetMarketView();
        }
      } catch (err) {
        setMarketError(err instanceof Error ? err.message : "删除历史记录失败");
      } finally {
        setMarketDeletingSessionId(null);
      }
    },
    [marketCurrentSession?.id, resetMarketView, token],
  );

  useEffect(() => {
    if (!token) {
      setSingleSessions([]);
      setMarketSessions([]);
      resetSingleView();
      resetMarketView();
      return;
    }
    void loadSingleSessions();
    void loadMarketSessions();
  }, [loadMarketSessions, loadSingleSessions, resetMarketView, resetSingleView, token]);

  const singleDisplayedMessages = useMemo(() => {
    if (!singleStreamingAssistant) {
      return singleMessages;
    }
    return [...singleMessages, { id: -1, role: "assistant", content: singleStreamingAssistant }];
  }, [singleMessages, singleStreamingAssistant]);

  const marketDisplayedMessages = useMemo(() => {
    if (!marketStreamingAssistant) {
      return marketMessages;
    }
    return [...marketMessages, { id: -1, role: "assistant", content: marketStreamingAssistant }];
  }, [marketMessages, marketStreamingAssistant]);

  const singleValue = useMemo<SingleAnalysisWorkspaceValue>(
    () => ({
      form: singleForm,
      setForm: setSingleForm,
      sessions: singleSessions,
      currentSession: singleCurrentSession,
      messages: singleMessages,
      displayedMessages: singleDisplayedMessages,
      result: singleResult,
      draft: singleDraft,
      setDraft: setSingleDraft,
      stages: singleStages,
      loadingSessions: singleLoadingSessions,
      deletingSessionId: singleDeletingSessionId,
      sending: singleSending,
      error: singleError,
      selectSession: selectSingleSession,
      resetView: resetSingleView,
      loadSessions: loadSingleSessions,
      sendStream: sendSingleStream,
      deleteSession: deleteSingleSession,
    }),
    [
      deleteSingleSession,
      loadSingleSessions,
      resetSingleView,
      selectSingleSession,
      sendSingleStream,
      setSingleForm,
      singleCurrentSession,
      singleDeletingSessionId,
      singleDisplayedMessages,
      singleDraft,
      singleError,
      singleForm,
      singleLoadingSessions,
      singleMessages,
      singleResult,
      singleSending,
      singleSessions,
      singleStages,
    ],
  );

  const marketValue = useMemo<MarketScanWorkspaceValue>(
    () => ({
      form: marketForm,
      setForm: setMarketForm,
      sessions: marketSessions,
      currentSession: marketCurrentSession,
      messages: marketMessages,
      displayedMessages: marketDisplayedMessages,
      result: marketResult,
      draft: marketDraft,
      setDraft: setMarketDraft,
      stages: marketStages,
      loadingSessions: marketLoadingSessions,
      deletingSessionId: marketDeletingSessionId,
      sending: marketSending,
      error: marketError,
      selectSession: selectMarketSession,
      resetView: resetMarketView,
      loadSessions: loadMarketSessions,
      sendStream: sendMarketStream,
      deleteSession: deleteMarketSession,
    }),
    [
      deleteMarketSession,
      loadMarketSessions,
      marketCurrentSession,
      marketDeletingSessionId,
      marketDisplayedMessages,
      marketDraft,
      marketError,
      marketForm,
      marketLoadingSessions,
      marketMessages,
      marketResult,
      marketSending,
      marketSessions,
      marketStages,
      resetMarketView,
      selectMarketSession,
      sendMarketStream,
      setMarketForm,
    ],
  );

  return (
    <SingleAnalysisWorkspaceContext.Provider value={singleValue}>
      <MarketScanWorkspaceContext.Provider value={marketValue}>{children}</MarketScanWorkspaceContext.Provider>
    </SingleAnalysisWorkspaceContext.Provider>
  );
}

export function useSingleAnalysisWorkspace() {
  const context = useContext(SingleAnalysisWorkspaceContext);
  if (!context) {
    throw new Error("useSingleAnalysisWorkspace must be used inside AnalysisWorkspaceProvider");
  }
  return context;
}

export function useMarketScanWorkspace() {
  const context = useContext(MarketScanWorkspaceContext);
  if (!context) {
    throw new Error("useMarketScanWorkspace must be used inside AnalysisWorkspaceProvider");
  }
  return context;
}

export { BASE_STAGES, buildStageState };
