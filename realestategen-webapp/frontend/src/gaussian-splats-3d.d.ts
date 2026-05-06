declare module '@mkkellogg/gaussian-splats-3d' {
  export enum WebXRMode { None = 0 }
  export enum RenderMode { Always = 0 }
  export enum SceneRevealMode { Gradual = 0 }

  export interface ViewerOptions {
    rootElement?: HTMLElement;
    selfDrivenMode?: boolean;
    sharedMemoryForWorkers?: boolean;
    dynamicScene?: boolean;
    webXRMode?: WebXRMode;
    renderMode?: RenderMode;
    sceneRevealMode?: SceneRevealMode;
    [key: string]: any;
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(url: string, options?: Record<string, any>): Promise<void>;
    start(): void;
    stop(): void;
    dispose(): void;
    renderer?: any;
    camera?: any;
    splatMesh?: any;
  }
}
