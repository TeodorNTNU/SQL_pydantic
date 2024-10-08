# Generated by Django 5.1 on 2024-08-28 20:10

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ElectricityData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('city', models.CharField(db_index=True, max_length=100)),
                ('state', models.CharField(db_index=True, max_length=100)),
                ('date', models.DateField(db_index=True)),
                ('region', models.CharField(db_index=True, max_length=100)),
                ('NOK_per_kWh', models.FloatField()),
                ('EUR_per_kWh', models.FloatField()),
                ('EXR', models.FloatField()),
                ('time_start', models.CharField(max_length=50)),
                ('time_end', models.CharField(max_length=50)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'electricity_prices',
            },
        ),
        migrations.CreateModel(
            name='WeatherData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('city', models.CharField(db_index=True, max_length=100)),
                ('state', models.CharField(db_index=True, max_length=100)),
                ('date', models.DateField(db_index=True)),
                ('temperature', models.FloatField()),
                ('time_start', models.CharField(max_length=50)),
                ('time_end', models.CharField(max_length=50)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'weather_data',
            },
        ),
    ]
