import pandas as pd
import numpy as np
import random
import csv
from datetime import datetime, timezone

solar_df = pd.read_csv('./MAET/Rl-agents/Data/select_data.csv')
solar_generation = np.array(solar_df["Pmpp FranceWatts panel (Wh)"], dtype=np.float32)

dates = solar_df["Date and time (UTC)"]

dates['datetime'] = pd.to_datetime(dates)
hour = dates['datetime'].dt.hour

"""
0 = night time
1 = day time
2 = peak time
3 = emergency time 
"""
classified_day = []

for i in range(len(hour)):
    hours = hour[i]
    if 6 <= hours < 18:
        class_name = 1
    elif 7 <= hours <= 9 or 17 <= hours <= 19:
        class_name = 2
    else:
        class_name = 0

    if 6 <= hours <= 19:
        if class_name == 3:
            pass
        else:
            if random.random() <= 0.01:

                class_name = 3

    classified_day.append(class_name)

    if class_name == 3:
        solar_df["Pmpp FranceWatts panel (Wh)"][i] = 0
solar_df["PV_class"] = classified_day

energy_consumption = np.array(solar_df["Total electricity consumption (Wh)"], dtype=np.float32)

classified_day_2 = []
for i in range(len(hour)):
    hours = hour[i]
    if random.random() <= 0.01:
        class_name = 1
    else:
        class_name = 0

    classified_day_2.append(class_name)

    if class_name == 1:
        solar_df["Total electricity consumption (Wh)"][i] = 0

solar_df["Consum_class"] = classified_day_2

solar_df.to_csv('MAET/Rl-agents/Data/select_data_v2.csv')
print("done")



