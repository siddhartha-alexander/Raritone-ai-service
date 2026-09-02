import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from raw_tryon import RawTryOnPipeline


app = FastAPI(
    title="Raritone 2D Virtual Try-On API",
    version="2.0.0"
)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "outputs" / "uploads"
RESULT_DIR = BASE_DIR / "outputs" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_TYPES = {
    "image/jpeg",
    "image/png"
}


print("Loading trained VTON pipeline...")

pipeline = RawTryOnPipeline()

print("Trained VTON pipeline loaded.")


@app.get("/api/ai/health")
def health():
    return {
        "success": True,
        "service": "raritone-vton",
        "status": "healthy",
        "model": "RaritoneVTONNet"
    }


@app.post("/api/ai/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    start_time = time.time()

    if person_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Person image must be JPG or PNG."
        )

    if garment_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Garment image must be JPG or PNG."
        )

    request_id = str(uuid.uuid4())

    request_upload_dir = (
        UPLOAD_DIR / request_id
    )

    request_result_dir = (
        RESULT_DIR / request_id
    )

    request_upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    request_result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    person_extension = (
        Path(person_image.filename).suffix
        or ".jpg"
    )

    garment_extension = (
        Path(garment_image.filename).suffix
        or ".jpg"
    )

    person_path = (
        request_upload_dir
        / f"person{person_extension}"
    )

    garment_path = (
        request_upload_dir
        / f"garment{garment_extension}"
    )

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    person_path.write_bytes(person_bytes)
    garment_path.write_bytes(garment_bytes)

    result_path = (
        request_result_dir
        / "tryon_result.jpg"
    )

    try:
        result = pipeline.run(
            person_image_path=str(person_path),
            garment_image_path=str(garment_path),
            output_path=str(result_path)
        )

    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Try-on processing failed: {str(error)}"
        )

    if not result.get("success"):
        raise HTTPException(
            status_code=422,
            detail=result.get(
                "error",
                "Try-on processing failed."
            )
        )

    processing_time = round(
        time.time() - start_time,
        3
    )

    return {
        "success": True,
        "request_id": request_id,
        "processing_time": processing_time,
        "model": "RaritoneVTONNet",
        "result": (
            f"/api/ai/tryon/result/{request_id}"
        )
    }


@app.get(
    "/api/ai/tryon/result/{request_id}"
)
def get_tryon_result(request_id: str):

    result_path = (
        RESULT_DIR
        / request_id
        / "tryon_result.jpg"
    )

    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Try-on result not found."
        )

    return FileResponse(
        result_path,
        media_type="image/jpeg",
        filename="tryon_result.jpg"
    )