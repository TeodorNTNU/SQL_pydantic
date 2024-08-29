from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import requests_cache
import openmeteo_requests
import pandas as pd
from functools import lru_cache
from retry_requests import retry

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from pydantic import BaseModel, Field

from django.conf import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)


Base = declarative_base()


# Data Models
class ElectricityData(Base):
    __tablename__ = 'electricity_prices'
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True)
    state = Column(String, index=True)
    date = Column(Date, index=True)
    region = Column(String, index=True)
    NOK_per_kWh = Column(Float)
    EUR_per_kWh = Column(Float)
    EXR = Column(Float)
    time_start = Column(String)
    time_end = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class WeatherData(Base):
    __tablename__ = 'weather_data'
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True)
    state = Column(String, index=True)
    date = Column(Date, index=True)
    temperature = Column(Float)
    time_start = Column(String)
    time_end = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# SQLAlchemy configuration
SQLALCHEMY_DATABASE_URL = settings.SQLALCHEMY_DATABASE_URL
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_recycle=settings.POOL_RECYCLE, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

def create_database():
    try:
        Base.metadata.create_all(bind=engine)
        logger.debug("Database and tables created.")
    except Exception as e:
        logger.error(f"Error creating database and tables: {e}")


def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset complete.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class ElPriceResponse(BaseModel):
    NOK_per_kWh: float
    EUR_per_kWh: float
    EXR: float
    time_start: str
    time_end: str

    class Config:
        from_attributes = True

class ElPriceRequest(BaseModel):
    city: str = Field(description="City")
    state: str = Field(description="State or region")
    date: str = Field(description="Date query ('YYYY-MM-DD/YYYY-MM-DD')")

class WeatherDataResponse(BaseModel):
    temperature: float
    time_start: str
    time_end: str

    class Config:
        from_attributes = True

class WeatherRequest(BaseModel):
    city: str = Field(description="City")
    state: str = Field(description="State or region")
    date: str = Field(description="Date query (YYYY-MM-DD/YYYY-MM-DD')")

# Utility Functions
def parse_date(date: str) -> Tuple[datetime, datetime]:
    try:
        if '/' in date:
            start_date_str, end_date_str = date.split('/')
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        else:
            start_date = end_date = datetime.strptime(date, '%Y-%m-%d')

        if start_date > end_date:
            raise ValueError("Start date cannot be after end date.")

        return start_date, end_date
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date}. Expected format is YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD. Error: {str(e)}")

# Data Tools
class BaseDataTool:
    geolocator = Nominatim(user_agent="data_tool_api")

    def __init__(self, db: Session):
        self.db = db

    @lru_cache(maxsize=100)
    def get_coordinates(self, city_name: str, state: str) -> Optional[Tuple[float, float]]:
        location = self.geolocator.geocode(f"{city_name}, {state}")
        return (location.latitude, location.longitude) if location else None

class ElectricityPriceTool(BaseDataTool):
    REGIONS = {
        "NO1": ("Oslo", (59.9139, 10.7522)),
        "NO2": ("Kristiansand", (58.1467, 7.9956)),
        "NO3": ("Trondheim", (63.4305, 10.3951)),
        "NO4": ("Tromsø", (69.6492, 18.9553)),
        "NO5": ("Bergen", (60.3928, 5.3221)),
    }

    session = requests_cache.CachedSession('electricity_cache', expire_after=86400)

    def find_nearest_region(self, lat: float, lon: float) -> Optional[str]:
        nearest_region = min(
            self.REGIONS.items(),
            key=lambda region: geodesic((lat, lon), region[1][1]).kilometers
        )[0]
        return nearest_region

    def fetch_electricity_prices(self, date: str, region: str) -> list:
        year, month, day = date.split('-')
        url = f"https://www.hvakosterstrommen.no/api/v1/prices/{year}/{month}-{day}_{region}.json"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error fetching data: {response.status_code}"}

    def store_prices(self, request: Dict[str, Any], region: str, prices: list):
        try:
            data = [
                {
                    "city": request['city'],
                    "state": request['state'],
                    "date": datetime.strptime(request['date'], '%Y-%m-%d').date(),
                    "region": region,
                    "NOK_per_kWh": price["NOK_per_kWh"],
                    "EUR_per_kWh": price["EUR_per_kWh"],
                    "EXR": price["EXR"],
                    "time_start": price["time_start"],
                    "time_end": price["time_end"]
                }
                for price in prices
            ]
            
            self.db.bulk_save_objects([ElectricityData(**entry) for entry in data])
            self.db.commit()
        except Exception as e:
            print(f"Error storing prices: {e}")
            self.db.rollback()

    def get_electricity_prices(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        coordinates = self.get_coordinates(request['city'], request['state'])
        if not coordinates:
            return {"error": "City not found"}

        lat, lon = coordinates
        start_date, end_date = parse_date(request['date'])
        nearest_region = self.find_nearest_region(lat, lon)
        all_prices = []

        if nearest_region:
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                prices = self.fetch_electricity_prices(date_str, nearest_region)
                if isinstance(prices, list):
                    self.store_prices({**request, 'date': date_str}, nearest_region, prices)
                    all_prices.extend(prices)
                current_date += timedelta(days=1)

            return all_prices if all_prices else {"error": "No data found for the given date range."}

        return {"error": "Could not determine the nearest region."}

class WeatherDataTool(BaseDataTool):
    cache_session = requests_cache.CachedSession('.weather_cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    def store_weather_data(self, request: Dict[str, Any], observations: list):
        data = [
            {
                "city": request['city'],
                "state": request['state'],
                "date": datetime.strptime(observation['date'], '%Y-%m-%d').date(),
                "temperature": observation['temperature'],
                "time_start": observation['time_start'],
                "time_end": observation['time_end'],
            }
            for observation in observations
        ]
        self.db.bulk_save_objects([WeatherData(**entry) for entry in data])
        self.db.commit()

    def get_weather_data(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        coordinates = self.get_coordinates(request['city'], request['state'])
        if not coordinates:
            return {"error": "Could not find coordinates for the specified city and state."}

        latitude, longitude = coordinates
        start_date, end_date = parse_date(request['date'])

        all_observations = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "temperature_2m"
            }
            response = self.openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]

            hourly = response.Hourly()
            hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

            hourly_dates = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ).strftime('%Y-%m-%dT%H:%M:%S%z')

            observations = [
                {
                    "date": date_str,
                    "temperature": float(hourly_temperature_2m[i]),
                    "time_start": hourly_dates[i],
                    "time_end": hourly_dates[i + 1]
                }
                for i in range(len(hourly_dates) - 1)
            ]

            all_observations.extend(observations)
            current_date += timedelta(days=1)

        self.store_weather_data(request, all_observations)

        return all_observations

# Main Functions for Querying Data
def get_all_electricity_prices(db: Session, electricity_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool = ElectricityPriceTool(db)
    return tool.get_electricity_prices(electricity_request)

def get_all_weather_data(db: Session, weather_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool = WeatherDataTool(db)
    return tool.get_weather_data(weather_request)


def process_electricity_prices(prices: List[ElPriceResponse]):
    if all(isinstance(price, dict) for price in prices):
        return prices
    else:
        return [price.dict() for price in prices]


def process_weather_data(data: List[WeatherDataResponse]) -> List[Dict[str, Any]]:
    if all(isinstance(item, dict) for item in data):
        return data
    else:    
        return [item.dict() for item in data]

def query_electricity_prices(db: Session, city: str, state: str, date: str) -> List[Dict[str, Any]]:
    prices = get_all_electricity_prices(db, {"city": city, "state": state, "date": date})
    # Convert to a list of dictionaries
    return process_electricity_prices(prices)

def query_weather_data(db: Session, city: str, state: str, date: str) -> List[Dict[str, Any]]:
    request = {"city": city, "state": state, "date": date}
    weather_data = get_all_weather_data(db, request)
    return process_weather_data(weather_data)

@tool
def electricity_price_tool(city: str, state: str, date: str) -> List[Dict[str, Any]]:
    """
    Fetches electricity prices for a given city, state, and date query.

    """
    with next(get_db()) as db:
        prices = query_electricity_prices(db, city, state, date)
    return prices

@tool
def weather_data_tool(city: str, state: str, date: str) -> List[Dict[str, Any]]:
    """
    Fetches weather data for a given city, state, and date query.

    """
    with next(get_db()) as db:
        weather_data = query_weather_data(db, city, state, date)
    return weather_data


# Fishing tools (data retrieval)
fishing_tools = [electricity_price_tool, weather_data_tool]

# %%
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

engine = create_engine(SQLALCHEMY_DATABASE_URL)
db = SQLDatabase(engine)

# Assuming `db` is your database session and `llm` is your language model
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# SQL tools (data analysis)
sql_tools = toolkit.get_tools()



tools = sql_tools + fishing_tools