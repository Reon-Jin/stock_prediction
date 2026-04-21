import { useEffect, useState } from "react";
import { BrainCircuit, Orbit, PanelTop } from "lucide-react";
import { PageTransition } from "../components/PageTransition";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import type { PlaceholderMap } from "../lib/types";

const iconMap = {
  decision_model: BrainCircuit,
  llm_summary: Orbit,
  more_features: PanelTop,
} as const;

export function PlaceholdersPage() {
  const { token } = useAuth();
  const [data, setData] = useState<PlaceholderMap | null>(null);

  useEffect(() => {
    if (!token) return;
    api.get<PlaceholderMap>("/platform/placeholders", token).then(setData).catch(() => undefined);
  }, [token]);

  return (
    <PageTransition>
      <section className="panel">
        <div className="panel-head">
          <h3>更多功能</h3>
          <span className="small-muted">后续能力将持续补充</span>
        </div>
        <div className="feature-grid">
          {data
            ? Object.entries(data).map(([key, value]) => {
                const Icon = iconMap[key as keyof typeof iconMap];
                return (
                  <article key={key} className="info-card dark">
                    <Icon size={24} />
                    <h4>{value.title}</h4>
                    <p>{value.description}</p>
                    <strong className="pill muted">{value.status}</strong>
                  </article>
                );
              })
            : null}
        </div>
      </section>
    </PageTransition>
  );
}
