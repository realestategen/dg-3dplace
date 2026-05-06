import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getSceneStatus, saveCapture, listCaptures, captureImageUrl, splatUrl, SceneStatus, Capture } from "../api";
import SplatViewer from "../components/SplatViewer";
import "./SceneViewerPage.css";

export default function SceneViewerPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const sceneId = Number(id);

  const [status, setStatus] = useState<SceneStatus | null>(null);
  const [captures, setCaptures] = useState<Capture[]>([]);
  const [capturing, setCapturing] = useState(false);
  const [lightboxId, setLightboxId] = useState<number | null>(null);
  const viewerRef = useRef<{ capture: () => string | null }>(null);

  // poll status until done or failed
  useEffect(() => {
    if (!sceneId) return;
    const poll = async () => {
      const s = await getSceneStatus(sceneId);
      setStatus(s);
      if (s.status === "done") {
        listCaptures(sceneId).then(setCaptures);
      }
    };
    poll();
    const iv = setInterval(async () => {
      const s = await getSceneStatus(sceneId);
      setStatus(s);
      if (s.status === "done" || s.status === "failed") clearInterval(iv);
    }, 5000);
    return () => clearInterval(iv);
  }, [sceneId]);

  async function handleCapture() {
    if (!viewerRef.current) return;
    const dataUrl = viewerRef.current.capture();
    if (!dataUrl) return;
    setCapturing(true);
    try {
      const c = await saveCapture(sceneId, dataUrl);
      setCaptures(prev => [c, ...prev]);
    } finally {
      setCapturing(false);
    }
  }

  if (!sceneId) {
    return <div className="svp-empty">Select a scene from the <a onClick={() => navigate("/scenes")} style={{cursor:"pointer"}}>Scenes list</a>.</div>;
  }

  return (
    <div className="svp">
      <div className="svp-header">
        <button className="btn-ghost svp-back" onClick={() => navigate("/scenes")}>← Back</button>
        <div className="svp-title">
          <span>Scene #{sceneId}</span>
          {status && (
            <span className={`svp-badge svp-badge--${status.status}`}>
              {status.status}
            </span>
          )}
        </div>
        {status?.status === "done" && (
          <button className="btn-primary" onClick={handleCapture} disabled={capturing}>
            {capturing ? "Saving…" : "📷 Capture Snapshot"}
          </button>
        )}
      </div>

      {status?.status === "done" ? (
        <div className="svp-viewer">
          <SplatViewer ref={viewerRef} url={splatUrl(sceneId)} />
        </div>
      ) : (
        <div className="svp-processing">
          <div className="svp-spinner" />
          <p className="svp-proc-label">
            {status ? `Processing… (${status.status})` : "Loading…"}
          </p>
          {status && status.log_tail.length > 0 && (
            <pre className="svp-log">{status.log_tail.join("\n")}</pre>
          )}
        </div>
      )}

      {captures.length > 0 && (
        <div className="svp-captures">
          <h3>Captured Snapshots</h3>
          <div className="svp-capture-grid">
            {captures.map(c => (
              <div key={c.id} className="svp-capture-thumb" onClick={() => setLightboxId(c.id)}>
                <img src={captureImageUrl(c.id)} alt={c.filename} />
                <span className="svp-capture-time">
                  {new Date(c.created_at).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {lightboxId !== null && (
        <div className="svp-lightbox" onClick={() => setLightboxId(null)}>
          <img src={captureImageUrl(lightboxId)} alt="capture" onClick={e => e.stopPropagation()} />
          <button className="svp-lb-close btn-ghost" onClick={() => setLightboxId(null)}>✕</button>
        </div>
      )}
    </div>
  );
}
