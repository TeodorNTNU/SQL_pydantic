# Generated by Django 5.1 on 2024-08-29 07:44

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chatapp', '0001_initial'),
    ]

    operations = [
        migrations.DeleteModel(
            name='ElectricityData',
        ),
        migrations.DeleteModel(
            name='WeatherData',
        ),
    ]
