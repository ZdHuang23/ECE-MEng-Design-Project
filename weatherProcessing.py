

import os
import numpy as np
import pandas as pd
from numpy import loadtxt
from math import exp, floor, pi
import pysolar
from datetime import datetime, timedelta, timezone
from pvlib.irradiance import get_total_irradiance
import requests
import csv

def skytempcalc(Temp,Cloud,RelaH):
    Tair = Temp + 273.15
    if Cloud == 0:
        Tdew = Temp - (100 - RelaH) / 5 # Emperical form which can be found here: The Relationship between RelativeHumidity and the DewpointTemperature in Moist Air
        esp_clear = 0.711 + 0.56 * (Tdew/100) + 0.73 * (Tdew/100)**2
        Tsky = Tair * esp_clear ** 0.25 - 273.15
    else:
        # This function is used as for calculating the sky temperature based on the paper
        # Comparative Study of Experimental and Theoretical Evaluation of Nocturnal Cooling System for Room Cooling for Clear and Cloudy Sky Climate"
        eps = (1 - 0.84 * Cloud) * (0.527 + 0.161 * exp(8.45 * (1 - 273 / Tair)) + 0.84 * Cloud)
        Tsky = Tair * eps ** 0.25 - 273.15
    return Tsky

class weatherProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def preprocess(self):
        # Read the input CSV file
        raw_dataframe = pd.read_csv(self.input_file).iloc[:,:]

        # Extracting relevant data from forecast
        temp = np.array(raw_dataframe['temp'])[:8760].reshape(-1,1)
        wind = np.array(raw_dataframe['windspeed'])[:8760].reshape(-1,1) / 3.6
        RH = np.array(raw_dataframe['humidity'])[:8760].reshape(-1,1)
        Cloud_ind = np.array(raw_dataframe['cloudcover'])[:8760].reshape(-1,1) / 100
        Tsky = np.zeros(len(temp)).reshape(-1,1)
        for i in range(len(temp)):
            Tsky[i] = skytempcalc(temp[i],Cloud_ind[i],RH[i])

        
        globalrad = raw_dataframe['dhi'].astype(float) # DHI
        IdH = raw_dataframe['dni'].astype(float) #DNI
        IbH = raw_dataframe['dewpoint'].astype(float) #Dew Point
        # print(IbH[0:10])

        start_t = datetime(2020, 3, 1, 0, 0, 0, 0, tzinfo=timezone.utc)

        latitude = 32.25  # Location of Desired position
        longitude = -110.91

        # Direct radiation
        NE_dir_wall = np.zeros(len(temp)).reshape(-1,1)
        NE_dir_roof = np.zeros(len(temp)).reshape(-1,1)
        SE_dir_wall = np.zeros(len(temp)).reshape(-1,1)
        SE_dir_roof = np.zeros(len(temp)).reshape(-1,1)
        NW_dir_wall = np.zeros(len(temp)).reshape(-1,1)
        NW_dir_roof = np.zeros(len(temp)).reshape(-1,1)
        SW_dir_wall = np.zeros(len(temp)).reshape(-1,1)
        SW_dir_roof = np.zeros(len(temp)).reshape(-1,1)

        # Diffuse radiation
        NE_dif_wall = np.zeros(len(temp)).reshape(-1,1)
        NE_dif_roof = np.zeros(len(temp)).reshape(-1,1)
        SE_dif_wall = np.zeros(len(temp)).reshape(-1,1)
        SE_dif_roof = np.zeros(len(temp)).reshape(-1,1)
        NW_dif_wall = np.zeros(len(temp)).reshape(-1,1)
        NW_dif_roof = np.zeros(len(temp)).reshape(-1,1)
        SW_dif_wall = np.zeros(len(temp)).reshape(-1,1)
        SW_dif_roof = np.zeros(len(temp)).reshape(-1,1)


        for i in range(len(temp)):
            sp_time = start_t + timedelta(hours = i)
            zenith = pysolar.solar.get_altitude(latitude, longitude, sp_time) 
            azimuth = pysolar.solar.get_azimuth(latitude, longitude, sp_time) 
            NE_dir_wall[i] = (get_total_irradiance(90,45,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            NE_dir_roof[i] = (get_total_irradiance(90,45,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            SE_dir_wall[i] = (get_total_irradiance(90,135,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            SE_dir_roof[i] = (get_total_irradiance(21.8,135,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            NW_dir_wall[i] = (get_total_irradiance(90,315,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            NE_dir_roof[i] = (get_total_irradiance(90,315,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            SW_dir_wall[i] = (get_total_irradiance(90,225,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            SW_dir_roof[i] = (get_total_irradiance(21.8,225,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_direct']
            
            NE_dif_wall[i] = (get_total_irradiance(90,45,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            NE_dif_roof[i] = (get_total_irradiance(90,45,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            SE_dif_wall[i] = (get_total_irradiance(90,135,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            SE_dif_roof[i] = (get_total_irradiance(21.8,135,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            NW_dif_wall[i] = (get_total_irradiance(90,315,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            NE_dif_roof[i] = (get_total_irradiance(90,315,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            SW_dif_wall[i] = (get_total_irradiance(90,225,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']
            SW_dif_roof[i] = (get_total_irradiance(21.8,225,zenith,azimuth,IbH[i],globalrad[i],IdH[i]))['poa_diffuse']

        # Save the processed data to a new CSV file
        indexnum = np.arange(1,len(temp)+1,1).reshape(-1,1)
        compiled_greenhouse = np.hstack([indexnum,temp,Tsky,wind,RH,NE_dir_wall,
                                        NE_dir_roof,SE_dir_wall,SE_dir_roof,
                                        NW_dir_wall,NW_dir_roof,SW_dir_wall,
                                        SW_dir_roof,NE_dif_wall,NE_dif_roof,
                                        SE_dif_wall,SE_dif_roof,NW_dif_wall,
                                        NW_dif_roof,SW_dif_wall,SW_dif_roof])
        np.savetxt(self.output_file, compiled_greenhouse, delimiter=",")

if __name__ == '__main__':
    input_file = 'ithacaweather2024.csv'
    output_file = 'processedithacaweather2024.csv'
    weather_processor = weatherProcessor(input_file, output_file)
    weather_processor.preprocess()
    print("Data processing completed")