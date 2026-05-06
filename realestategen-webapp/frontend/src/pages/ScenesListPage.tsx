import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listScenes, Scene } from "../api";
import "./ScenesListPage.css";

const STATUS_COLORS: Record<string, string> = {
  pending: "#8892a4",
  processing_frames: "#f59e0b",
  training: "#6c63ff",
  exporting: "#a78bfa",
  done: "#22c55e",
  failed: "#ef4444",
};

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending",
  processing_frames: "Processing Frames",
  training: "Training",
  exporting: "Exporting",
  done: "Ready",
  failed: "Failed",
};

export default function ScenesListPage() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    listScenes().then(setScenes).finally(() => setLoading(false));
    const iv = setInterval(() => listScenes().then(setScenes), 5000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="sl">
      <div className="sl-header">
        <h1>Scenes</h1>
        <p className="sl-sub">Click a ready scene to open the interactive 3DGS viewer</p>
      </div>

      {loading ? (
        <div className="sl-loading"><span className="sl-spinner" /></div>
      ) : scenes.length === 0 ? (
        <p className="sl-empty">No scenes yet. Go to Videos to create one.</p>
      ) : (
        <div className="sl-list">
          {scenes.map(s => (
            <div
              key={s.id}
              className={`sl-row ${s.status === "done" ? "clickable" : ""}`}
              onClick={() => s.status === "done" && navigate(`/scenes/${s.id}`)}
            >
              <div className="sl-row-left">
                <span className="sl-name">{s.name}</span>
                <span className="sl-date">{new Date(s.created_at).toLocaleString()}</span>
              </div>
              <span
                className="sl-status"
                style={{ color: STATUS_COLORS[s.status] ?? "#8892a4" }}
              >
                {s.status !== "done" && s.status !== "failed" && (
                  <span className="sl-spin" />
                )}
                {STATUS_LABELS[s.status] ?? s.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
