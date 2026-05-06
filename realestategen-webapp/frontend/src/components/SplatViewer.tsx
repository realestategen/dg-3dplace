import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";

interface Props {
  url: string;
}

export interface SplatViewerHandle {
  capture: () => string | null;
}

const SplatViewer = forwardRef<SplatViewerHandle, Props>(({ url }, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useImperativeHandle(ref, () => ({
    capture() {
      const renderer = rendererRef.current;
      if (!renderer) return null;
      renderer.render(
        viewerRef.current?.splatMesh?.scene ?? new THREE.Scene(),
        viewerRef.current?.camera ?? new THREE.PerspectiveCamera()
      );
      return renderer.domElement.toDataURL("image/png");
    },
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    const viewer = new GaussianSplats3D.Viewer({
      rootElement: containerRef.current,
      selfDrivenMode: true,
      sharedMemoryForWorkers: false,
      dynamicScene: false,
      webXRMode: GaussianSplats3D.WebXRMode.None,
      renderMode: GaussianSplats3D.RenderMode.Always,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Gradual,
    });

    viewerRef.current = viewer;

    // grab renderer reference for canvas capture
    if ((viewer as any).renderer) {
      rendererRef.current = (viewer as any).renderer;
    }

    viewer
      .addSplatScene(url, { splatAlphaRemovalThreshold: 5 })
      .then(() => {
        viewer.start();
        // try to get renderer after start
        if ((viewer as any).renderer) {
          rendererRef.current = (viewer as any).renderer;
        }
      })
      .catch(console.error);

    return () => {
      try {
        viewer.stop?.();
        viewer.dispose?.();
      } catch (_) {}
    };
  }, [url]);

  // override canvas capture: grab the actual canvas element
  useImperativeHandle(ref, () => ({
    capture() {
      const canvas = containerRef.current?.querySelector("canvas");
      if (!canvas) return null;
      return canvas.toDataURL("image/png");
    },
  }));

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", background: "#000" }}
    />
  );
});

SplatViewer.displayName = "SplatViewer";
export default SplatViewer;
