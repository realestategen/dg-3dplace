const BASE = "http://localhost:8765";

export interface Video {
  id: number;
  name: string;
  filename: string;
  size: number;
  has_thumbnail: boolean;
  created_at: string;
}

export interface Scene {
  id: number;
  video_id: number;
  name: string;
  status: string;
  splat_path: string | null;
  created_at: string;
  updated_at: string;
}

export interface SceneStatus {
  id: number;
  status: string;
  log_tail: string[];
  splat_path: string | null;
}

export interface Capture {
  id: number;
  scene_id: number;
  filename: string;
  created_at: string;
}

export async function uploadVideo(file: File): Promise<Video> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/videos`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteVideo(id: number): Promise<void> {
  await fetch(`${BASE}/api/videos/${id}`, { method: "DELETE" });
}

export async function listVideos(): Promise<Video[]> {
  const res = await fetch(`${BASE}/api/videos`);
  return res.json();
}

export function videoStreamUrl(id: number) {
  return `${BASE}/api/videos/${id}/stream`;
}

export function videoThumbnailUrl(id: number) {
  return `${BASE}/api/videos/${id}/thumbnail`;
}

export async function createScene(video_id: number, name: string): Promise<Scene> {
  const res = await fetch(`${BASE}/api/scenes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id, name }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listScenes(): Promise<Scene[]> {
  const res = await fetch(`${BASE}/api/scenes`);
  return res.json();
}

export async function getSceneStatus(id: number): Promise<SceneStatus> {
  const res = await fetch(`${BASE}/api/scenes/${id}/status`);
  return res.json();
}

export function splatUrl(scene_id: number) {
  return `${BASE}/scenes/${scene_id}/output/splat_export/splat.ply`;
}

export async function saveCapture(scene_id: number, image_data: string): Promise<Capture> {
  const res = await fetch(`${BASE}/api/captures`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id, image_data }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listCaptures(scene_id: number): Promise<Capture[]> {
  const res = await fetch(`${BASE}/api/captures/scene/${scene_id}`);
  return res.json();
}

export function captureImageUrl(capture_id: number) {
  return `${BASE}/api/captures/${capture_id}/image`;
}
