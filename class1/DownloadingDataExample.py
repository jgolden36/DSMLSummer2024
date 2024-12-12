import os
import http.client
import urllib.parse
import pandas as pd
from time import sleep
from datetime import datetime, timedelta

# Function to fetch data for a specific day and startRow
def fetch_data_for_day(base_url, api_path, date, start_row, row_count=50000):
    params = urllib.parse.urlencode({
        'rowCount': row_count,
        'startRow': start_row,
        'datetime_beginning_ept': f"{date.strftime('%m-%d-%Y')} 00:00 to {date.strftime('%m-%d-%Y')} 23:59",
        'format': 'csv'
    })
    headers = {'Ocp-Apim-Subscription-Key': 'your-API-Key',
        'Content-Type': 'application/json',
    }
    conn = http.client.HTTPSConnection(base_url)
    sleep(10)
    conn.request("GET", f"{api_path}?{params}", "", headers)
    response = conn.getresponse()
    if response.status != 200:
        raise Exception(f"HTTP Error: {response.status} {response.reason}")
    data = response.read()
    print(data)
    conn.close()
    return data.decode('utf-8')

# Function to collect data for a single day with pagination
def collect_data_for_day(base_url, api_path, date):
    all_data = []
    start_row = 1
    first_chunk = True
    while True:
        day_data = fetch_data_for_day(base_url, api_path, date, start_row)
        if first_chunk:
            # If it's the first chunk, include the header
            all_data.append(day_data)
            first_chunk = False
        else:
            # Skip the header in subsequent chunks
            all_data.append(day_data.split('\n', 1)[1])
        # Check if the received data is less than the rowCount, meaning no more data left
        if len(day_data.splitlines()) < 50001:  # 50000 data rows + 1 header row
            break
        start_row += 50000
    return ''.join(all_data)

# Define the main function to iterate over date range and save data
def main(base_url, api_path, start_date, end_date, output_file):
    current_date = start_date
    with open(output_file, 'a',encoding="utf-8") as f:
        while current_date <= end_date:
            print(f"Fetching data for {current_date.strftime('%Y-%m-%d')}")
            day_data = collect_data_for_day(base_url, api_path, current_date)
            f.write(day_data)
            current_date += timedelta(days=1)

# Define the API URL and date range
base_url = 'api.pjm.com'
api_path = '/api/v1/da_hrl_lmps'
start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 12, 31)
output_file = 'api_data.csv'

# Run the main function
main(base_url, api_path, start_date, end_date, output_file)

start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 12, 31)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\FuelType')
output_file = 'FuelTypeData.csv'
api_path = '/api/v1/gen_by_fuel'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Flows')
output_file = 'FlowData.csv'
api_path = '/api/v1/da_interface_flows_and_limits'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Losses')
output_file = 'Losses.csv'
api_path = '/api/v1/gen_ehv_losses'
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Constraints')
output_file = 'ConstraintData.csv'
api_path = '/api/v1/da_transconstraints'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\SolarGenbyArea')
output_file = 'SolarGen.csv'
api_path = '/api/v1/hourly_solar_power_forecast'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\WindGenbyArea')
output_file = 'WindGen.csv'
api_path = '/api/v1/hourly_wind_power_forecast'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Load Forecast')
output_file = 'ForecastedLoad.csv'
api_path = '/api/v1/load_frcstd_hist'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Transfer Limits')
output_file = 'TransferLimits.csv'
api_path = '/api/v1/transfer_limits_and_flows'
main(base_url, api_path, start_date, end_date, output_file)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Actual_Schedule')
output_file = 'ActualInterchange.csv'
api_path = '/api/v1/act_sch_interchange'
main(base_url, api_path, start_date, end_date, output_file)
start_date = datetime(2017, 4, 26)
end_date = datetime(2023, 12, 31)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\State Interchange')
output_file = 'ActualInterchange.csv'
api_path = '/api/v1/act_sch_interchange'
main(base_url, api_path, start_date, end_date, output_file)
start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 12, 31)
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Classes Fall 2023\\Electricity Markets\\HVDC Project Grant\\Data\\PJM Data\\Tie Flows')
output_file = 'DCTieFlows.csv'
api_path = '/api/v1/rt_scheduled_interchange'
main(base_url, api_path, start_date, end_date, output_file)
