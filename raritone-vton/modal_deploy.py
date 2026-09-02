from pathlib import Path
import sys

import modal


APP_NAME = "raritone-vton"

LOCAL_DIR = Path(__file__).resolve().parent

REMOTE_DIR = "/root/raritone-vton"


image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libgomp1"
    )
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "numpy",
        "Pillow",
        "opencv-python-headless",
        "torch",
        "torchvision",
        "transformers",
        "safetensors",
        "rembg",
        "onnxruntime",
    )
    .add_local_dir(
        LOCAL_DIR,
        REMOTE_DIR,
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            "outputs/**",
            "dataset/**",
            "evaluation/**",
            "kaggle/**",
            "tests/**",
        ],
    )
)


app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    memory=8192,
)
@modal.asgi_app()
def fastapi_app():

    sys.path.insert(
        0,
        f"{REMOTE_DIR}/app"
    )

    # main.py expects BASE_DIR to resolve correctly
    sys.path.insert(
        0,
        REMOTE_DIR
    )

    from main import app as api

    return api