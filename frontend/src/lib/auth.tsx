import {
  PropsWithChildren,
  createContext,
  startTransition,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { api } from "./api";
import type { AuthResponse, User } from "./types";

type AuthContextValue = {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (account: string, password: string) => Promise<void>;
  register: (payload: { username: string; email: string; password: string; display_name?: string }) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
};

const TOKEN_KEY = "stock_ui_token";
const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: PropsWithChildren) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const saveAuth = (response: AuthResponse) => {
    localStorage.setItem(TOKEN_KEY, response.access_token);
    startTransition(() => {
      setToken(response.access_token);
      setUser(response.user);
    });
  };

  const refresh = async () => {
    const currentToken = localStorage.getItem(TOKEN_KEY);
    if (!currentToken) {
      setLoading(false);
      return;
    }
    try {
      const me = await api.get<User>("/auth/me", currentToken);
      setToken(currentToken);
      setUser(me);
    } catch {
      localStorage.removeItem(TOKEN_KEY);
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      loading,
      login: async (account, password) => {
        const response = await api.post<AuthResponse>("/auth/login", { account, password }, null, 8000);
        saveAuth(response);
      },
      register: async (payload) => {
        const response = await api.post<AuthResponse>("/auth/register", payload);
        saveAuth(response);
      },
      logout: () => {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setUser(null);
      },
      refresh,
    }),
    [loading, token, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}
