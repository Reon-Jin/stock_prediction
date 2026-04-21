export type User = {
  id: number;
  username: string;
  email: string;
  display_name?: string | null;
  is_active: boolean;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  user: User;
};

export type DashboardSummary = {
  latest_trade_date: string | null;
  training_sample_count: number;
  security_count: number;
  active_user_count: number;
  latest_checkpoint: string | null;
  latest_checkpoint_time: string | null;
  current_modules: Array<{ key: string; title: string; status: string }>;
  planned_modules: Array<{ key: string; title: string; status: string }>;
};

export type ModelRun = {
  run_name: string;
  checkpoint_path: string | null;
  updated_at: string | null;
  train_minutes: number | null;
  best_valid_loss: number | null;
  test_p_win_acc: number | null;
  test_rank_score_mae: number | null;
  has_test_metrics: boolean;
};

export type DecisionAction = {
  action: string;
  action_cn: string;
  confidence: number;
  suggested_hold_days: number;
  best_horizon: string;
  path: string;
  priority: number;
};

export type RiskReview = {
  passed: boolean;
  original_action: string;
  final_action: string;
  downgraded: boolean;
  risk_flags: string[];
  risk_warnings: string[];
  blocked_rules: string[];
};

export type DecisionResult = {
  symbol: string;
  symbol_name: string;
  trade_date: string;
  decision: DecisionAction;
  scores: {
    S_final: number;
    S_3: number;
    S_5: number;
    S_10: number;
    S_20: number;
    S_40: number;
    consistency: number;
    conflict_state?: string | null;
    R_score?: number;
  };
  model_output: Record<string, number>;
  market_regime: string;
  risk_flags: string[];
  risk_review: RiskReview;
  reasons: string[];
  metadata: Record<string, string>;
};

export type SingleAnalysisResult = {
  analysis_date: string;
  effective_trade_date: string;
  checkpoint_path: string | null;
  engine_version: string;
  model_info?: {
    source: string;
    descriptor: string;
    checkpoint_path: string | null;
    feature_version: string;
  };
  stock: {
    symbol: string;
    name: string;
    industry_sw: string;
    board: string;
  };
  holding_context: {
    is_holding: boolean;
    holding_days: number;
    used_by_decision_model: boolean;
    risk_preference: string;
  };
  sample?: {
    format: string;
    meta: {
      symbol: string;
      trade_date: string;
      name: string;
      industry_sw: string;
      board: string;
    };
    X_seq: number[][];
    X_tab: number[];
    X_event: number[];
    X_mkt: number[];
    X_company_ids: {
      symbol_id: number;
      industry_id: number;
      board_id: number;
    };
    X_company_profile: number[];
    neighbors: {
      neighbor_symbol_ids: number[];
      neighbor_scores: number[];
    };
    schema: {
      seq_length: number;
      neighbor_topk: number;
      seq_columns: string[];
      tab_columns: string[];
      event_columns: string[];
      mkt_columns: string[];
      company_id_columns: string[];
      company_profile_columns: string[];
    };
  };
  company_info?: Record<string, number>;
  prediction: Record<string, number>;
  market_snapshot: Record<string, number>;
  decision_result: DecisionResult;
};

export type MarketCandidate = {
  rank: number;
  symbol: string;
  name: string;
  industry_sw: string;
  board: string;
  close: number;
  pct_chg: number;
  avg_amount_5: number;
  recommended_hold_days?: number;
  recommended_hold_label?: string;
  predicted_win_rate?: number;
  signal_score?: number;
  rank_score_pred?: number;
  ret_mu_pred?: number;
  risk_dd_pred?: number;
  bigloss_prob?: number;
  market_regime_prob?: number;
  feature_missing_rate?: number;
  company_info?: Record<string, number>;
  prediction?: Record<string, number>;
  market_snapshot?: Record<string, number>;
  decision_result: DecisionResult;
  risk_flags: string[];
  reasons: string[];
};

export type MarketScanResult = {
  analysis_date?: string;
  effective_trade_date: string;
  checkpoint_path: string | null;
  engine_version: string;
  scan_mode?: "market" | "quick" | string;
  sample_size?: number;
  market_total_candidates?: number;
  total_candidates: number;
  top_n: number;
  pool_size: number;
  selected_count: number;
  market_regime_counts: Record<string, number>;
  candidates: MarketCandidate[];
};

export type PlaceholderMap = {
  decision_model: { status: string; title: string; description: string };
  llm_summary: { status: string; title: string; description: string };
  more_features: { status: string; title: string; description: string };
};

export type AnalysisMessage = {
  id: number;
  role: "user" | "assistant" | string;
  content: string;
  created_at?: string | null;
};

export type AnalysisSessionSummary = {
  id: number;
  symbol: string;
  stock_name?: string | null;
  title?: string | null;
  latest_trade_date?: string | null;
  is_holding: boolean;
  holding_days: number;
  risk_preference: string;
  last_user_message?: string | null;
  last_assistant_message?: string | null;
  updated_at?: string | null;
};

export type AnalysisSessionDetail = AnalysisSessionSummary & {
  messages: AnalysisMessage[];
  latest_analysis?: SingleAnalysisResult | null;
};

export type AnalysisStageKey = "candidate_select" | "extract_data" | "model_predict" | "decision_engine" | "llm_analysis";

export type AnalysisStage = {
  stage: AnalysisStageKey;
  label: string;
  status: "idle" | "running" | "done";
  current?: number;
  total?: number;
  message?: string;
};
