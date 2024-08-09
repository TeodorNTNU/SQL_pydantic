from fastapi import FastAPI, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session
from typing import List
import models
from database import engine, SessionLocal
from schemas import ElectricityPriceRequest, ElectricityPriceCreate
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Electricity Price API"}


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint to read all prices
@app.get('/prices', response_model=List[ElectricityPriceCreate])
async def read_all_prices(db: Session = Depends(get_db)):
    return db.query(models.ElectricityPrice).all()

# Endpoint to create a new price entry
@app.post('/price', status_code=status.HTTP_201_CREATED, response_model=ElectricityPriceCreate)
async def create_price(price_request: ElectricityPriceRequest, db: Session = Depends(get_db)):
    coordinates = get_coordinates(price_request.city, price_request.state)
    if not coordinates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="City not found")
    
    lat, lon = coordinates
    nearest_region = find_nearest_region(lat, lon)
    if not nearest_region:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not determine the nearest region")

    prices = fetch_electricity_prices(price_request.date_query, nearest_region)
    if isinstance(prices, str):  # If there's an error message
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=prices)

    for price in prices:
        db_price = models.ElectricityPrice(
            city=price_request.city,
            state=price_request.state,
            date_query=price_request.date_query,
            region=nearest_region,
            NOK_per_kWh=price["NOK_per_kWh"],
            EUR_per_kWh=price["EUR_per_kWh"],
            EXR=price["EXR"],
            time_start=price["time_start"],
            time_end=price["time_end"]
        )
        db.add(db_price)
    db.commit()
    return db_price

# Utility functions to get coordinates and fetch electricity prices
def get_coordinates(city_name: str, state: str) -> tuple:
    geolocator = Nominatim(user_agent="electricity_price_api")
    location = geolocator.geocode(f"{city_name}, {state}")
    if location:
        return location.latitude, location.longitude
    else:
        return None

def find_nearest_region(lat: float, lon: float) -> str:
    REGIONS = {
        "NO1": ("Oslo", (59.9139, 10.7522)),
        "NO2": ("Kristiansand", (58.1467, 7.9956)),
        "NO3": ("Trondheim", (63.4305, 10.3951)),
        "NO4": ("Tromsø", (69.6492, 18.9553)),
        "NO5": ("Bergen", (60.3928, 5.3221)),
    }
    min_distance = float("inf")
    nearest_region = None
    for region, (city, coordinates) in REGIONS.items():
        distance = geodesic((lat, lon), coordinates).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_region = region
    return nearest_region

def fetch_electricity_prices(date_query: str, region: str) -> list:
    # Assume date_query is formatted as "YYYY-MM-DD"
    year, month, day = date_query.split("-")
    url = f"https://www.hvakosterstrommen.no/api/v1/prices/{year}/{month}-{day}_{region}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching data: {response.status_code}"
