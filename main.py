import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import CONFIDENCE_THRESHOLD, UPLOAD_DIR
from database import DocumentRecord, get_db
from inference import extract_fields
from model_loader import get_model
from schemas import DocumentResponse, ExtractedData, StatisticsResponse, VerificationRequest

processor, model = get_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def _build_document_response(record: DocumentRecord) -> DocumentResponse:
    extracted_data = None
    if record.extracted_json:
        extracted_data = ExtractedData(**json.loads(record.extracted_json))

    human_corrections = None
    if record.human_corrections:
        human_corrections = json.loads(record.human_corrections)

    return DocumentResponse(
        id=record.id,
        filename=record.filename,
        upload_path=record.upload_path,
        status=record.status,
        extracted_data=extracted_data,
        overall_confidence=record.overall_confidence,
        human_corrections=human_corrections,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@app.post("/documents/upload", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename).suffix or ".png"
    unique_name = f"{uuid.uuid4()}{ext}"
    save_path = Path(UPLOAD_DIR) / unique_name

    with open(save_path, "wb") as f:
        f.write(file.file.read())

    result = extract_fields(processor, model, str(save_path))

    status = "approved" if result["overall_confidence"] >= CONFIDENCE_THRESHOLD else "review"

    record = DocumentRecord(
        filename=file.filename,
        upload_path=str(save_path),
        status=status,
        extracted_json=json.dumps(result),
        overall_confidence=result["overall_confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return _build_document_response(record)


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    response: Response,
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(DocumentRecord)
    if status:
        query = query.filter(DocumentRecord.status == status)
    total = query.count()
    records = (
        query.order_by(DocumentRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    response.headers["X-Total-Count"] = str(total)
    return [_build_document_response(r) for r in records]


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return _build_document_response(record)


@app.put("/documents/{doc_id}/verify", response_model=DocumentResponse)
def verify_document(doc_id: int, body: VerificationRequest, db: Session = Depends(get_db)):
    record = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    record.human_corrections = json.dumps(body.corrected_data)
    record.status = "verified"
    record.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(record)
    return _build_document_response(record)


@app.get("/statistics", response_model=StatisticsResponse)
def get_statistics(db: Session = Depends(get_db)):
    total = db.query(DocumentRecord).count()

    rows = (
        db.query(DocumentRecord.status, func.count(DocumentRecord.id))
        .group_by(DocumentRecord.status)
        .all()
    )
    by_status = {status: count for status, count in rows}

    avg_conf = (
        db.query(func.avg(DocumentRecord.overall_confidence))
        .filter(DocumentRecord.overall_confidence.isnot(None))
        .scalar()
    )

    return StatisticsResponse(
        total_documents=total,
        by_status=by_status,
        average_confidence=avg_conf,
    )
