from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./receipts.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DocumentRecord(Base):
    __tablename__ = "document_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    upload_path = Column(String, nullable=False)
    status = Column(String, default="pending")
    extracted_json = Column(Text)
    overall_confidence = Column(Float)
    human_corrections = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Base.metadata.create_all(bind=engine)
