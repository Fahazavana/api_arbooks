from sqlalchemy import Column, Integer, String, LargeBinary
from .db import Base


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    embedding = Column(LargeBinary)
