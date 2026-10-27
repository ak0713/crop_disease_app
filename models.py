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
