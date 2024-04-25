import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./Data/select_data_v2.csv')
df['Date and time (UTC)'] = pd.to_datetime(df['Date and time (UTC)'])
df['hour'] = df['Date and time (UTC)'].dt.hour
agent0 = df.iloc[0::2]
agent0.data = agent0.values
agent0.data_time = agent0['hour'].values

agent1 = df.iloc[1::2]
agent1.data = agent1.values
agent1.data_time = agent1['hour'].values

plt.plot(agent0.data_time[:144], agent0['Total electricity consumption (Wh)'][:144]/1000, color='g', label='consumption')
plt.plot(agent0.data_time[:144], agent1['Pmpp FranceWatts panel (Wh)'][:144], color='r', label='generation')
plt.xlabel('Time (hours)')
plt.ylabel('Total electricity consumption/generation (kWh)')
plt.legend()
plt.show()
