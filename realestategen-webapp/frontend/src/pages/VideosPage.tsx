import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listVideos, uploadVideo, createScene, deleteVideo, videoStreamUrl, videoThumbnailUrl, Video } from "../api";
import "./VideosPage.css";

export default function VideosPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<Video | null>(null);
  const [creating, setCreating] = useState<number | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadingRef = useRef(false);
  const navigate = useNavigate();

  useEffect(() => {
    listVideos().then(setVideos).finally(() => setLoading(false));
  }, []);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    if (uploadingRef.current) return;
    uploadingRef.current = true;
    setUploading(true);
    // reset input so re-selecting the same file works, and prevents onChange double-fire on drop
    if (inputRef.current) inputRef.current.value = "";
    try {
      for (const file of Array.from(files)) {
        const v = await uploadVideo(file);
        setVideos(prev => [v, ...prev]);
      }
    } finally {
      setUploading(false);
      uploadingRef.current = false;
    }
  }

  async function handleCreateScene(video: Video) {
    setCreating(video.id);
    try {
      const scene = await createScene(video.id, `${video.name}_scene`);
      navigate(`/scenes/${scene.id}`);
    } finally {
      setCreating(null);
    }
  }

  async function handleDelete(video: Video, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Remove "${video.name}"?`)) return;
    setDeleting(video.id);
    try {
      await deleteVideo(video.id);
      setVideos(prev => prev.filter(v => v.id !== video.id));
      if (preview?.id === video.id) setPreview(null);
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="vp">
      <div className="vp-header">
        <h1>Videos</h1>
        <p className="vp-sub">Upload MP4 videos to create 3D Gaussian Splatting scenes</p>
      </div>

      <div
        className={`vp-drop ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); e.stopPropagation(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      >
        <input ref={inputRef} type="file" accept="video/*" multiple hidden
          onChange={e => handleFiles(e.target.files)} />
        {uploading ? (
          <span className="vp-drop-text">Uploading…</span>
        ) : (
          <>
            <span className="vp-drop-icon">⬆</span>
            <span className="vp-drop-text">Drop video here or click to browse</span>
            <span className="vp-drop-sub">MP4, MOV, AVI, MKV</span>
          </>
        )}
      </div>

      {loading ? (
        <div className="vp-loading"><span className="vp-spinner" /></div>
      ) : videos.length === 0 ? (
        <p className="vp-empty">No videos yet. Upload one above.</p>
      ) : (
        <div className="vp-grid">
          {videos.map(v => (
            <div key={v.id} className="vp-card">
              <div className="vp-thumb" onClick={() => setPreview(v)}>
                {v.has_thumbnail
                  ? <img src={videoThumbnailUrl(v.id)} alt={v.name} />
                  : <div className="vp-thumb-placeholder" />}
                <span className="vp-play">▶</span>
              </div>
              <div className="vp-info">
                <span className="vp-name">{v.name}</span>
                <span className="vp-meta">{(v.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
              <div className="vp-actions">
                <button className="btn-ghost" onClick={() => setPreview(v)}>Preview</button>
                <button
                  className="btn-primary"
                  onClick={() => handleCreateScene(v)}
                  disabled={creating === v.id}
                >
                  {creating === v.id ? "Starting…" : "Create 3DGS"}
                </button>
                <button
                  className="btn-danger vp-del"
                  onClick={e => handleDelete(v, e)}
                  disabled={deleting === v.id}
                  title="Remove video"
                >
                  {deleting === v.id ? "…" : "✕"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {preview && (
        <div className="vp-modal" onClick={() => setPreview(null)}>
          <div className="vp-modal-inner" onClick={e => e.stopPropagation()}>
            <div className="vp-modal-header">
              <span>{preview.name}</span>
              <button className="btn-ghost" onClick={() => setPreview(null)}>✕</button>
            </div>
            <video
              src={videoStreamUrl(preview.id)}
              controls
              autoPlay
              style={{ width: "100%", borderRadius: 8 }}
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button
                className="btn-primary"
                onClick={() => { setPreview(null); handleCreateScene(preview); }}
                disabled={creating === preview.id}
              >
                {creating === preview.id ? "Starting…" : "Create 3DGS Scene"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
