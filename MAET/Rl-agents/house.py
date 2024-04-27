import gym

from gym import spaces

import argparse
import yaml

import numpy as np


import pandas

import pickle


def registration(max_episode):
    gym.envs.register(id='Progressus-v1',

                      entry_point='house:ProgressusEnv',

                      max_episode_steps=max_episode,

                      kwargs={

                          'global_seed': None,

                          'agent_id': None,

                          'n_agents': None

                      })
def ConfigObjectProduce():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default="./Config_house/Progressus.yml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class ProgressusEnv(gym.Env):
    def __init__(self, global_seed=None, agent_id=None, n_agents=None):
        # Simulation Section
        self.global_seed = global_seed
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.data_time = 0
        self.sell_price = .0
        self.action_space = spaces.MultiDiscrete([5,4,2])
        # self.action_space = spaces.Discrete(4)
        self.config = ConfigObjectProduce()
  
        # load data (csv)
        df = pandas.read_csv(self.config['simulation']['envTrain'], sep=",", header=0)
        df['Date and time (UTC)'] = pandas.to_datetime(df['Date and time (UTC)'])
        df['hour'] = df['Date and time (UTC)'].dt.hour
        data = df.iloc[self.agent_id::self.n_agents]
        self.data = data.values
        self.data_time = data['hour'].values
        # �29.66 per 100 kilowatt-hour
        # �0.0002966 per watt-hour
        self.sell_price = (.3 * (1 - np.exp(-((self.data_time - 14) ** 2) / 5))) * 1e-3
        if self.agent_id == 0:
            with open('Time.pkl', 'ab') as f:
                pickle.dump(self.data_time, f)
            with open('price.pkl', 'ab') as f:
                pickle.dump(self.sell_price, f)
        self.panelProdMax = max(self.data[:, 5]) 
        self.consumptionMax = max(self.data[:, 4])
        self.priceMax = max(abs(self.data[:, 3]))
        self.ECPVMax = max(self.data[:,6])
        self.ECCMax = max(self.data[:,7])
        # print("max price", self.priceMax)
        self.data[:, 5] /= 1 # in kW. production
        self.data[:, 4] /= 1000  # in kW. consumption
        # self.data[:, 3] /= self.priceMax
        self.data[:, 3] /= 100  # in euros per kWh
        self.currentState_row = 0
        self.currentState_price = self.data[self.currentState_row, 3] # in euros per kWh
        self.currentState_consumption = self.data[self.currentState_row, 4] # in kw
        self.currentState_panelProd = self.data[self.currentState_row, 5]# in kw
        self.currentState_ECPV = self.data[self.currentState_row, 6]
        self.currentState_ECC = self.data[self.currentState_row, 7]

        self.currentState_battery = 0.0
        self.diffProd = 0
        # Capacity of the battery
        self.batteryCapacity = 2 # kWh
        # CO2 price/emissions
        self.co2Price = 25.0 * 0.001  # price per ton of CO2 (mean price from the european market)
        self.co2Generator = 8 * 0.001  # kg of CO2 generated per kWh from the diesel generator
        self.co2Market = (
            0.3204  # kg of CO2 generated per kWh from the national power market (danish)
        )
        # Operational Rewards
        self.chargingReward = 0.0
        self.dischargingReward = 0.0
        # self.solarRewards = 0.0
        self.generatorReward = 0.0  # 0.314 � 0.528 $/kWh
        # Yields
        self.chargingYield = 1.0
        self.dischargingYield = 1.0
        self.state = None

              # please, keep the lase element obs space element devoted to track agents prb allocation
        OBS_SPACE_DIM = 3
        high = np.array((self.batteryCapacity, self.panelProdMax, self.consumptionMax/1000, 24, self.ECPVMax, self.ECCMax), dtype=np.float32)
        low = np.array((0, 0, 0, 0, 0, 0), dtype=np.float32)
        # low = np.zeros((OBS_SPACE_DIM,), dtype=np.float32)
        # low = np.ones((OBS_SPACE_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # self.reset()
        # self.seed()

    def load_data(self):
        return self.data, self.data_time

    def step(self, action):
        #  map action to the corresponding string
        self.Action_grid = action[0]
        self.Action_emergency_PV = action[1]
        self.Action_emergency_Consum = action[2]

        if self.Action_grid == 0:
            Action_grid_agent = "charge"
        elif self.Action_grid == 1:
            Action_grid_agent = "charge_sell"
        elif self.Action_grid == 2:
            Action_grid_agent = "discharge"
        elif self.Action_grid == 3:
            Action_grid_agent = "sell"
        elif self.Action_grid == 4:
            Action_grid_agent = "buy"

        #  compute tje energy surplus
        self.diffProd = self.currentState_panelProd - self.currentState_consumption
        #  compute the temporary battery level. This value can be negative or exceed the battery capacity
        current_battery_temp = self.currentState_battery + (action[0] - 3) * (action[0] - 4) * self.diffProd

        #  define the relu function
        def relu(x):
            return np.maximum(0, x)

        #  define the reward
        # reward_general = -10*relu(-(current_battery_temp-.1*self.batteryCapacity))
        #  reward for charging the battery. It penalizes the agent if the battery is charged too much, over the battery capacity
        reward_charge = -relu(current_battery_temp - self.batteryCapacity) * (action[0] - 1) * (action[0] - 2) * (
                    action[0] - 3) * (action[0] - 4)
        #  reward for charging the battery and selling the surplus energy
        reward_charge_sell = relu(current_battery_temp - self.batteryCapacity) * self.sell_price[
            self.currentState_row] * (
                                 action[0]) * (action[0] - 2) * (action[0] - 3) * (action[0] - 4)
        # reward for discharging the battery. It penalizes the agent if the battery is discharged too much, below 10% of the battery capacity
        #  Add constraint 10 to make sure that the battery is not discharged too much
        reward_discharge = -10 * relu(-(current_battery_temp - .1 * self.batteryCapacity)) * (action[0]) * (action[0] - 1) * (
                    action[0] - 3) * (
                                   action[0] - 4)
        #  reward for selling the surplus energy with selling price
        reward_sell = self.sell_price[self.currentState_row] * relu(self.diffProd) * (action[0]) * (action[0] - 1) * (
                    action[0] - 2) * (action[0] - 4)
        #  reward for buying energy from the grid. It has a negative sign because it is a cost
        reward_buy = -self.currentState_price * relu(-self.diffProd) * (action[0]) * (action[0] - 1) * (action[0] - 2) * (
                    action[0] - 3)
        reward_Emergency_response_PV = -10 * relu(abs(int(self.currentState_ECPV) - int(self.Action_emergency_PV)))
        reward_Emergency_response_Consum = -10 * relu(abs(int(self.currentState_ECC) - int(self.Action_emergency_Consum)))
        #  sum all the rewards
        reward = reward_charge + reward_charge_sell + reward_discharge + reward_sell + reward_buy + reward_Emergency_response_PV + reward_Emergency_response_Consum
        #  clip the current battery between 0 and the capacity. This is the real battery level
        self.currentState_battery = np.clip(current_battery_temp, 0, self.batteryCapacity)

        self.currentState_row += 1
        row = self.currentState_row

        self.currentState_price = self.data[row, 3]
        self.currentState_consumption = self.data[row, 4]
        self.currentState_panelProd = self.data[row, 5]
        self.currentState_ECPV = self.data[row, 6]
        self.currentState_ECC = self.data[row, 7]

        state1 = self.currentState_battery
        state2 = self.currentState_panelProd
        state3 = self.currentState_consumption
        state4 = self.data_time[self.currentState_row - 1]
        state5 = self.currentState_ECPV
        state6 = self.currentState_ECC

        co2_kg = relu(-self.diffProd) * 0.3855535

        self.state = (state1, state2, state3, state4, state5, state6)

        return np.array(self.state, dtype=np.float32), reward, False, dict(dic_diffpro=co2_kg,
                                                                                  dic_battery=self.currentState_battery)

    def reset(self):
        self.currentState_row = 0
        self.state = [self.currentState_battery, self.currentState_panelProd, self.currentState_consumption,
                      self.data_time[self.currentState_row - 1], self.currentState_ECPV, self.currentState_ECC]

        return np.array(self.state, dtype=np.float32)



