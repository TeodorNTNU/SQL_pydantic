# main.ipynb

# Imports
import json
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, ElectricityPrice
from schemas import ElectricityPriceRequest, PriceData
from typing import Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime, timedelta
import requests

# Create the database tables
Base.metadata.create_all(bind=engine)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Regions dictionary
REGIONS = {
    "NO1": ("Oslo", (59.9139, 10.7522)),
    "NO2": ("Kristiansand", (58.1467, 7.9956)),
    "NO3": ("Trondheim", (63.4305, 10.3951)),
    "NO4": ("Tromsø", (69.6492, 18.9553)),
    "NO5": ("Bergen", (60.3928, 5.3221)),
}

# Functions
def get_coordinates(city_name: str, state: str) -> Optional[tuple]:
    geolocator = Nominatim(user_agent="electricity_price_api")
    location = geolocator.geocode(f"{city_name}, {state}")
    if location:
        return location.latitude, location.longitude
    else:
        return None

def find_nearest_region(lat: float, lon: float) -> Optional[str]:
    min_distance = float("inf")
    nearest_region = None
    for region, (city, coordinates) in REGIONS.items():
        distance = geodesic((lat, lon), coordinates).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_region = region
    return nearest_region

def fetch_electricity_prices(year: str, month: str, day: str, region: str) -> list:
    url = f"https://www.hvakosterstrommen.no/api/v1/prices/{year}/{month}-{day}_{region}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching data: {response.status_code}"

def store_prices(db: Session, request: ElectricityPriceRequest, region: str, prices: list):
    for price in prices:
        db_price = ElectricityPrice(
            city=request.city,
            state=request.state,
            date_query=request.date_query,
            region=region,
            NOK_per_kWh=price["NOK_per_kWh"],
            EUR_per_kWh=price["EUR_per_kWh"],
            EXR=price["EXR"],
            time_start=price["time_start"],
            time_end=price["time_end"]
        )
        db.add(db_price)
    db.commit()

def get_electricity_prices(request: ElectricityPriceRequest):
    coordinates = get_coordinates(request.city, request.state)
    if not coordinates:
        return {"error": "City not found"}
    
    lat, lon = coordinates
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    if request.date_query.lower() == "today":
        start_date = end_date
    elif request.date_query.lower() == "yesterday":
        start_date = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
        end_date = start_date
    elif "last" in request.date_query.lower():
        if "10 days" in request.date_query.lower():
            start_date = (datetime.now() - timedelta(10)).strftime("%Y-%m-%d")
        elif "30 days" in request.date_query.lower():
            start_date = (datetime.now() - timedelta(30)).strftime("%Y-%m-%d")
        elif "three months" in request.date_query.lower():
            start_date = (datetime.now() - timedelta(90)).strftime("%Y-%m-%d")
        else:
            return {"error": "Invalid date query"}
    else:
        try:
            start_date, end_date = request.date_query.split('/')
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD or predefined ranges."}

    nearest_region = find_nearest_region(lat, lon)
    if nearest_region:
        prices = fetch_electricity_prices(start_date[:4], start_date[5:7], start_date[8:10], nearest_region)
        if isinstance(prices, list):  # Ensure prices is a list of dictionaries
            db = next(get_db())
            store_prices(db, request, nearest_region, prices)
            return prices
        else:
            return prices  # Return the error message
    else:
        return "Could not determine the nearest region."

# Example usage
request = ElectricityPriceRequest(city="Oslo", state="Oslo", date_query="today")
result = get_electricity_prices(request)
print(result)
