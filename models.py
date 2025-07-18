from sqlalchemy import Column, Integer, String
from database import Base

class TopicSummary(Base):
    __tablename__ = "topic_summaries"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    summary = Column(String)
