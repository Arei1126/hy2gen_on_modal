import modal
import os

# 1. モデルのキャッシュを保存するVolumeを作成（これで再ダウンロードを防いで節約！）
#cache_volume = modal.Volume.from_name("hf-cache-v2", create_if_missing=True)
cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# 2. TRELLISと依存パッケージをインストールするImageを定義
image = (
        modal.Image.from_registry(
                "nvidia/cuda:12.9.2-devel-ubuntu22.04",
                add_python="3.10",
        )
        .apt_install(
                "git",
                "build-essential",
                "ninja-build",
                "libgl1-mesa-glx",
                "libglib2.0-0",
                "libegl1",
                "libglvnd0",
                "libglvnd-dev",
                "libgles2",
        )
        # 2. CUDAバージョンに適合するPyTorchを先にインストール
        .pip_install(
                "torch",
                "torchvision",
                "torchaudio",
                extra_index_url="https://download.pytorch.org/whl/cu121",
        )
        # 3. submodulesごとクローン
        .run_commands(
                #"git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive",
                "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
                )
        # 4. setup.sh を実行（Condaは使わず、既存のPython環境に直接入れる）
        # ※ --new-env は外して実行する
        .run_commands(
                 "pip install -r Hunyuan3D-2/requirements.txt",
                 "pip install -e ./Hunyuan3D-2",
        )
        .env({
            "HF_HOME": "/root/.cache/huggingface",
            #"HY3DGEN_HOME": "/root/.cache", # hy3dgenにも同じ場所を使わせる！
            })
)

app = modal.App(name="trellis-3d-generator")

# グローバル変数でパイプラインをキャッシュ（コンテナが生きてる間は再読み込みしない！）
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        print("Loading TRELLIS pipeline...")
        #_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2',subfolder='hunyuan3d-dit-v2-0',variant='fp16') #v2
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1',subfolder='hunyuan3d-dit-v2-1',use_safetensors=False,) #v2.1
        #_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2mini',subfolder='hunyuan3d-dit-v2-mini',variant='fp16') #v2mini

        cache_volume.commit()
    return _pipeline

@app.function(
    image=image,
    #gpu="L40S", # キミの予算に合わせて "L4" とか "A100" に変えてね！
    gpu="L4", # キミの予算に合わせて "L4" とか "A100" に変えてね！
    volumes={"/root/.cache/": cache_volume}, # キャッシュVolumeをマウント
    timeout=600,
)
def generate_3d_model(image_bytes: bytes) -> bytes:
    import sys

    #sys.path.append("/TRELLIS.2")

    # ここでTRELLISの推論コードを呼び出し
    print("TRELLIS2 loaded successfully!")
    import torch
    import tempfile
    
    cache_volume.reload()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    
    pipeline = get_pipeline()
    
    # 画像を一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        image_path = f.name
    
    try:
        print(f"Generating 3D model from {image_path}...")
        # 3Dモデルの生成（gaussian, radiance_field, mesh が含まれる dict が返ってくるよ）
        outputs = pipeline(image=image_path,
                           num_inference_steps=30,
                           octree_resolution=160,
                           #octree_resolution=184,
                           #num_chunks=20000,
                           num_chunks=20000,
                           #generator=torch.manual_seed(4234),
                           output_type='trimesh',
                           mc_algo='mc',
                           mc_level=-1/512,
                           guidance_scale = 5, #7.5
                           dual_guidance_scale = 14,# 10.5,
                           dual_guidance = True,
                           )

        mesh = outputs[0]

        # GLBファイルとしてエクスポート
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            glb_path = f.name
        mesh.export(glb_path)
        
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()
        return glb_bytes
    finally:
        # お片付け
        os.unlink(image_path)
        if 'glb_path' in locals():
            os.unlink(glb_path)

@app.local_entrypoint()
def main(image_path: str, output_path: str = "output.glb"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    print(f"Uploading image and starting generation...")
    # remote() でModalのクラウド上で実行！
    glb_bytes = generate_3d_model.remote(image_bytes)
    
    with open(output_path, "wb") as f:
        f.write(glb_bytes)
    
    print(f"✨ 3D model saved to {output_path}")
