import os
import sys
import argparse


def _resolve_hunyuan_paths() -> tuple[str, str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    hy_dir = os.path.join(os.path.dirname(here), "Hunyuan3D-2.1")
    hy_shape = os.path.join(hy_dir, "hy3dshape")
    hy_paint = os.path.join(hy_dir, "hy3dpaint")
    return hy_dir, hy_shape, hy_paint


def _setup_hunyuan_imports() -> tuple[str, str]:
    hy_dir, hy_shape, hy_paint = _resolve_hunyuan_paths()
    for p in [hy_dir, hy_shape, hy_paint]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return hy_dir, hy_paint


def run_paint(mesh_path: str, image_path: str, output_mesh_path: str, views: int, resolution: int) -> str:
    hy_dir, hy_paint = _setup_hunyuan_imports()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(max_num_view=views, resolution=resolution)

    realesrgan_ckpt = os.path.join(hy_paint, "ckpt", "RealESRGAN_x4plus.pth")
    multiview_cfg = os.path.join(hy_paint, "cfgs", "hunyuan-paint-pbr.yaml")
    custom_pipeline = os.path.join(hy_paint, "hunyuanpaintpbr")

    if os.path.exists(realesrgan_ckpt):
        conf.realesrgan_ckpt_path = realesrgan_ckpt
    if os.path.exists(multiview_cfg):
        conf.multiview_cfg_path = multiview_cfg
    if os.path.exists(custom_pipeline):
        conf.custom_pipeline = custom_pipeline

    pipeline = Hunyuan3DPaintPipeline(conf)
    out_path = pipeline(mesh_path=mesh_path, image_path=image_path, output_mesh_path=output_mesh_path)
    if out_path is None:
        out_path = output_mesh_path
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Run Hunyuan paint in isolated env")
    p.add_argument("--mesh", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--resolution", type=int, default=512)
    return p.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(args.mesh):
        raise FileNotFoundError(f"Mesh not found: {args.mesh}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    out = run_paint(
        mesh_path=args.mesh,
        image_path=args.image,
        output_mesh_path=args.output,
        views=max(6, min(9, int(args.views))),
        resolution=int(args.resolution),
    )
    print(out)


if __name__ == "__main__":
    main()
