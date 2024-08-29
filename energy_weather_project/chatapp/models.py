from django_sorcery.db import databases
import datetime

# Get the default database connection
db = databases.get("default")

# Define ElectricityData model
class ElectricityData(db.Model):
    id = db.Column(db.Integer, primary_key=True, index=True)
    city = db.Column(db.String, index=True)
    state = db.Column(db.String, index=True)
    date = db.Column(db.Date, index=True)
    region = db.Column(db.String, index=True)
    NOK_per_kWh = db.Column(db.Float)
    EUR_per_kWh = db.Column(db.Float)
    EXR = db.Column(db.Float)
    time_start = db.Column(db.String)
    time_end = db.Column(db.String)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Define WeatherData model
class WeatherData(db.Model):
    id = db.Column(db.Integer, primary_key=True, index=True)
    city = db.Column(db.String, index=True)
    state = db.Column(db.String, index=True)
    date = db.Column(db.Date, index=True)
    temperature = db.Column(db.Float)
    time_start = db.Column(db.String)
    time_end = db.Column(db.String)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
