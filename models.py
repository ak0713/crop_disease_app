from sqlalchemy import Column, Integer, String, Float
from database import Base

class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True)
    
    # ML Data
    disease_detected = Column(String)
    confidence_score = Column(Float)
    
    # Weather Data
    temperature_c = Column(Float)
    humidity_percent = Column(Float)
    soil_moisture = Column(Float)
    
    # Actionable Output
    action_warning = Column(String)

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
import datetime
from database import Base

class CommunityPost(Base):
    __tablename__ = "community_posts"
    id = Column(Integer, primary_key=True, index=True)
    farmer_name = Column(String)
    disease_tag = Column(String)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    images = relationship("PostImage", back_populates="post")
    comments = relationship("PostComment", back_populates="post")

class PostImage(Base):
    __tablename__ = "post_images"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("community_posts.id"))
    image_path = Column(String)
    post = relationship("CommunityPost", back_populates="images")

class PostComment(Base):
    __tablename__ = "post_comments"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("community_posts.id"))
    author = Column(String)
    text = Column(String)
    post = relationship("CommunityPost", back_populates="comments")