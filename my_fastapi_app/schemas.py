# schemas.py
from pydantic import BaseModel, Field
from typing import List

class PriceData(BaseModel):
    NOK_per_kWh: float
    EUR_per_kWh: float
    EXR: float
    time_start: str
    time_end: str

class ElectricityPriceRequest(BaseModel):
    city: str = Field(description="City")
    state: str = Field(description="State or region")
    date_query: str = Field(description="Date query")
