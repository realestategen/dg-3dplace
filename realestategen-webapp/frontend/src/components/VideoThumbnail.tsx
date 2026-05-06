import { useEffect, useRef, useState } from "react";

interface Props {
  src: string;
  onClick?: () => void;
}

export default function VideoThumbnail({ src, onClick }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [thumb, setThumb] = useState<string | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const capture = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")!.drawImage(video, 0, 0);
      setThumb(canvas.toDataURL("image/jpeg", 0.8));
    };

    const onSeeked = () => capture();
    const onLoaded = () => { video.currentTime = 1; };

    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("seeked", onSeeked);
    return () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("seeked", onSeeked);
    };
  }, [src]);

  return (
    <div className="vp-thumb" onClick={onClick}>
      <video ref={videoRef} src={src} muted preload="metadata" style={{ display: "none" }} />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {thumb ? (
        <img src={thumb} alt="thumbnail" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      ) : (
        <div className="vp-thumb-placeholder" />
      )}
      <span className="vp-play">▶</span>
    </div>
  );
}
