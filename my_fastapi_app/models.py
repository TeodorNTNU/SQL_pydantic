# models.py
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from database import Base
from datetime import datetime

class ElectricityPrice(Base):
    __tablename__ = 'electricity_prices'

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True)
    state = Column(String, index=True)
    date_query = Column(String)
    region = Column(String)
    NOK_per_kWh = Column(Float)
    EUR_per_kWh = Column(Float)
    EXR = Column(Float)
    time_start = Column(Text)
    time_end = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
