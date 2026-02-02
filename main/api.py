import io
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile # type: ignore
from fastapi.responses import JSONResponse, StreamingResponse # type: ignore

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from core.logger import Logger
from main.infer import infer

app = FastAPI(title="Rental Churn Inference API")


def _read_parquet_from_bytes(content: bytes) -> pd.DataFrame:
    if not content:
        raise HTTPException(status_code=400, detail="Empty parquet file.")

    try:
        table = pq.read_table(io.BytesIO(content))
        return table.to_pandas(ignore_metadata=True, strings_to_categorical=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read parquet: {exc}") from exc


def _save_parquet_to_default_location(content: bytes) -> str:
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    input_path = os.path.join(data_dir, "inference.parquet")
    with open(input_path, "wb") as handle:
        handle.write(content)
    return input_path


@app.post("/predict")
async def predict(
    run_id: str      = Form(..., description="Model/run ID"),
    batch_size: int  = Form(256, description="Batch size for inference"),
    file: UploadFile = File(..., description="Parquet file for inference"),
):
    if file.content_type not in {"application/octet-stream", "application/x-parquet", "application/vnd.apache.parquet"}:
        return JSONResponse(status_code=400, content={"detail": "Invalid file type. Please upload a parquet."})

    logger = Logger(name="api", level="INFO", log_dir=None)
    logger.info(f"[API] Received file: {file.filename}")

    content = await file.read()
    _read_parquet_from_bytes(content)
    _save_parquet_to_default_location(content)

    try:
        predictions, _ = infer(
            run_id=run_id,
            save_predictions=True,
            batch_size=batch_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    filename = f"predictions_{run_id}.csv"
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
