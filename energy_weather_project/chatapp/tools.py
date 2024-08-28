from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from chatapp.models import ElectricityData, WeatherData

from django.db import connection

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)

# Define your Pydantic models
class ElPriceResponse(BaseModel):
    NOK_per_kWh: float
    EUR_per_kWh: float
    EXR: float
    time_start: str
    time_end: str

    class Config:
        orm_mode = True

class WeatherDataResponse(BaseModel):
    temperature: float
    time_start: str
    time_end: str

    class Config:
        orm_mode = True

# Define tools for fetching and storing data using Django ORM
class BaseDataTool:
    geolocator = Nominatim(user_agent="data_tool_api")

    def get_coordinates(self, city_name: str, state: str) -> Optional[Tuple[float, float]]:
        location = self.geolocator.geocode(f"{city_name}, {state}")
        return (location.latitude, location.longitude) if location else None

class ElectricityPriceTool(BaseDataTool):
    session = requests_cache.CachedSession('electricity_cache', expire_after=86400)

    def fetch_electricity_prices(self, date: str, region: str) -> list:
        year, month, day = date.split('-')
        url = f"https://www.hvakosterstrommen.no/api/v1/prices/{year}/{month}-{day}_{region}.json"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error fetching data: {response.status_code}"}

    def store_prices(self, request: Dict[str, Any], region: str, prices: list):
        data = [
            ElectricityData(
                city=request['city'],
                state=request['state'],
                date=datetime.strptime(request['date'], '%Y-%m-%d').date(),
                region=region,
                NOK_per_kWh=price["NOK_per_kWh"],
                EUR_per_kWh=price["EUR_per_kWh"],
                EXR=price["EXR"],
                time_start=price["time_start"],
                time_end=price["time_end"]
            )
            for price in prices
        ]
        ElectricityData.objects.bulk_create(data)

    def get_electricity_prices(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Fetch data from API if not present in database
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
            WeatherData(
                city=request['city'],
                state=request['state'],
                date=datetime.strptime(observation['date'], '%Y-%m-%d').date(),
                temperature=observation['temperature'],
                time_start=observation['time_start'],
                time_end=observation['time_end']
            )
            for observation in observations
        ]
        WeatherData.objects.bulk_create(data)

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

# Updated example requests using Django ORM tools
def get_all_electricity_prices(electricity_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool = ElectricityPriceTool()
    return tool.get_electricity_prices(electricity_request)

def get_all_weather_data(weather_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool = WeatherDataTool()
    return tool.get_weather_data(weather_request)

# Update the tool functions to use the new request methods
@tool
def electricity_price_tool(city: str, state: str, date: str) -> List[Dict[str, Any]]:
    """
    Fetches electricity prices for a given city, state, and date query.
    """
    request = {"city": city, "state": state, "date": date}
    prices = get_all_electricity_prices(request)
    return prices

@tool
def weather_data_tool(city: str, state: str, date: str) -> List[Dict[str, Any]]:
    """
    Fetches weather data for a given city, state, and date query.
    """
    request = {"city": city, "state": state, "date": date}
    weather_data = get_all_weather_data(request)
    return weather_data

# Combine tools
tools = [electricity_price_tool, weather_data_tool]

# Example usage
if __name__ == "__main__":
    # Example request for electricity prices
    electricity_request = {
        "city": "Lillehammer",
        "state": "Oslo",
        "date": "2024-01-01/2024-10-02"
    }

    # Retrieve all electricity prices with automatic API fetching if necessary
    all_prices = get_all_electricity_prices(electricity_request)
    print(all_prices)

    # Example request for weather data
    weather_request = {
        "city": "Lillehammer",
        "state": "Lillehammer",
        "date": "2023-07-23/2023-07-25"
    }

    # Retrieve all weather data with automatic API fetching if necessary
    all_weather = get_all_weather_data(weather_request)
    print(all_weather)


# Fishing tools (data retrieval)
fishing_tools = [electricity_price_tool, weather_data_tool]


db = SQLDatabase(connection)  # Use Django's connection directly
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()


tools = sql_tools + fishing_tools