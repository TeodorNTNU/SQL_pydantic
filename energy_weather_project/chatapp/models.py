from django.db import models

from django.db import models

class ElectricityData(models.Model):
    city = models.CharField(max_length=100, db_index=True)
    state = models.CharField(max_length=100, db_index=True)
    date = models.DateField(db_index=True)
    region = models.CharField(max_length=100, db_index=True)
    NOK_per_kWh = models.FloatField()
    EUR_per_kWh = models.FloatField()
    EXR = models.FloatField()
    time_start = models.CharField(max_length=50)
    time_end = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'electricity_prices'


class WeatherData(models.Model):
    city = models.CharField(max_length=100, db_index=True)
    state = models.CharField(max_length=100, db_index=True)
    date = models.DateField(db_index=True)
    temperature = models.FloatField()
    time_start = models.CharField(max_length=50)
    time_end = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'weather_data'

