import requests
import csv
import datetime
class weatherData:
    def __init__(self, latitude=-33.856784, longitude=151.215297,output_file="weather_data.csv",days=7):
        self.latitude = latitude
        self.longitude = longitude
        self.output_file = output_file
        self.days = days
        self.YOUR_API_KEY = "YOUR_API_KEY"
    def getWeatherData(self):
        # Calculate the current datetime - 14 days
        current_datetime = datetime.datetime.now()
        start_datetime = current_datetime - datetime.timedelta(days=self.days+7)
        # Format the start datetime in the required format
        start_datetime_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # Create the URL to get the weather data
        url = f"https://api.solcast.com.au/data/historic/radiation_and_weather?latitude={self.latitude}" \
            f"&longitude={self.longitude}&start=2022-10-25T14:45:00.000Z&duration=P{self.days}D&period=PT60M&time_zone=utc" \
            f"&output_parameters=air_temp,wind_speed_10m,relative_humidity,cloud_opacity,dni,dhi,dewpoint_temp" \
            f"&api_key={self.YOUR_API_KEY}" \
            f"&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            estimated_actuals = data['estimated_actuals']
            if estimated_actuals:
                with open(self.output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['temp', 'windspeed', 'humidity', 'cloudcover', 'dhi', 'dni', 'dewpoint'])
                    for entry in estimated_actuals:
                        writer.writerow([entry['air_temp'],
                                         entry['wind_speed_10m'],
                                         entry['relative_humidity'],
                                         entry['cloud_opacity'],
                                         entry['dhi'],
                                         entry['dni'],
                                         entry['dewpoint_temp']])
                return True
            else:
                print("No historic data available")
                return False
        else:
            print(f"Error in getting data {response}")
            return False

if __name__ == '__main__':
    weather = weatherData(-33.856784, 151.215297)
    weather.getWeatherData()