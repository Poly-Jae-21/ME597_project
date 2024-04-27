#CUDA_VISIBLE_DEVICES=0 python3 MAET.py --config Config_house/Progressus.properties
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import configparser
import argparse
import gym
from house import registration
from Agent import DSAC
from MultiAgent import Multi_Agent
import time
import yaml
import matplotlib.pyplot as plt



def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default="./Config_house/Progressus.yml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    global_seed = config['simulation']['global_seed']
    n_house = int(config['simulation']['n_house'])
    n_decisions_per_episode = int(config['DRL']['Max_Episode'])
    n_fed_episodes = int(config['DRL']['n_episodes'])

    registration(n_decisions_per_episode)

    ENV_dic = {}
    AGENTS_dic = {}
    score_log = {}
    Reward_list=[]
    Total_Reward_list_agent0 = []
    Average_Reward_list_agent0 = []
    Load_Data_list_agent0 = []
    Load_Data_time_list_agent0 = []
    Emergency_response_PV_agent0 = []
    Emergency_response_Consum_agent0 = []
    battery_capacity_agent0 = []
    co2_agent0 = []

    Total_Reward_list_agent1 = []
    Average_Reward_list_agent1 = []
    Load_Data_list_agent1 = []
    Load_Data_time_list_agent1 = []
    Emergency_response_PV_agent1 = []
    Emergency_response_Consum_agent1 = []
    battery_capacity_agent1 = []
    co2_agent1 = []

    episodes_list=[]
    drl_class_NAME=''
    if n_house > 1:
        federation_mode = True
    else:
        federation_mode = False
    Reward_list_fed = []
    diffpro_list_fed = []
    battery_list_fed = []
    # Instantiation of Gym environments for every House
   
    if config['DRL']['_class_ML'] == 'DSAC':
        drl_class = DSAC
        drl_class_NAME = 'DSAC'
 




    for i in range(n_house):
        key = 'env' + str(i + 1)
        if key not in ENV_dic.keys():
            ENV_dic[key] = gym.make('Progressus-v1',
                                    global_seed=global_seed,
                                    agent_id=i,
                                    n_agents=n_house
                                    )
            AGENTS_dic[key] = Multi_Agent(drl_class, ENV_dic[key], config, i)


    # Train the Agents for n_episodes
    for j in range(int(config['DRL']['n_episodes'])):
        # save the model

        print("Episode = ",j,"/", config['DRL']['n_episodes'])

        # Multiprocessing
#        with ProcessPoolExecutor(10) as executor:
        for i in range(n_house):
            #p_process[i] = executor.submit(AGENTS_dic[('env' + str(i + 1))].run())
            load_data, load_data_time, total_rewards, avg_reward_per_house, pridction_PV, prediction_Consum, action_log, state_log, diffpro_log, battery_log, Emergency_response_PV, Emergency_response_Consum, best_score, best_diffpro, best_battery = AGENTS_dic[('env' + str(i + 1))].run()
            print(Emergency_response_PV, "%")
            if i == 0:
                Total_Reward_list_agent0.append(total_rewards)
                Average_Reward_list_agent0.append(avg_reward_per_house)
                Load_Data_list_agent0 = load_data
                Load_Data_time_list_agent0 = load_data_time
                Emergency_response_PV_agent0.append(Emergency_response_PV)
                Emergency_response_Consum_agent0.append(Emergency_response_Consum)
                battery_capacity_agent0.append(best_battery)
                co2_agent0.append(best_diffpro)

            else:
                Total_Reward_list_agent1.append(total_rewards)
                Average_Reward_list_agent1.append(avg_reward_per_house)
                Load_Data_list_agent1 = load_data
                Load_Data_time_list_agent1 = load_data_time
                Emergency_response_PV_agent1.append(Emergency_response_PV)
                Emergency_response_Consum_agent1.append(Emergency_response_Consum)
                battery_capacity_agent1.append(best_battery)
                co2_agent1.append(best_diffpro)


    end = time.time()
    simulation_time = end-start

    print("#####################################################################################")
    print("####################### Simulation done in : ", simulation_time)
    print("#####################################################################################")
    plt.plot(range(1,int(config['DRL']['n_episodes'])+1),Total_Reward_list_agent0)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards agent0')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

    plt.plot(range(1,int(config['DRL']['n_episodes'])+1), Emergency_response_PV_agent0, color='red', label='Emergency response_PV_agent0')
    plt.plot(range(1,int(config['DRL']['n_episodes'])+1), Emergency_response_Consum_agent0, color='blue', label='Emergency_response_Consum_agent0')
    plt.xlabel('Episodes')
    plt.ylabel('Probability of Emergency detection (%) in agent0')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.legend()
    plt.show()

    plt.plot(range(1, int(config['DRL']['n_episodes']) + 1), battery_capacity_agent0)
    plt.xlabel('Episodes')
    plt.ylabel('battery capacity agent0 (kWh)')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

    plt.plot(range(1, int(config['DRL']['n_episodes']) + 1), co2_agent0)
    plt.xlabel('Episodes')
    plt.ylabel('co2 emission by agent0 (kg)')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

    plt.plot(range(1,int(config['DRL']['n_episodes'])+1), Total_Reward_list_agent1)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards agent1')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

    plt.plot(range(1,int(config['DRL']['n_episodes'])+1), Emergency_response_PV_agent1, color='red',
             label='Emergency response_PV_agent1')
    plt.plot(range(1,int(config['DRL']['n_episodes'])+1), Emergency_response_Consum_agent1, color='blue',
             label='Emergency_response_Consum_agent1')
    plt.xlabel('Episodes')
    plt.ylabel('Probability of Emergency detection (%) in agent1')
    plt.legend()
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

    plt.plot(range(1, int(config['DRL']['n_episodes']) + 1), battery_capacity_agent1)
    plt.xlabel('Episodes')
    plt.ylabel('battery capacity agent1 (kWh)')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

    plt.plot(range(1, int(config['DRL']['n_episodes']) + 1), co2_agent1)
    plt.xlabel('Episodes')
    plt.ylabel('co2 emission by agent1 (kg)')
    plt.xlim(1, int(config['DRL']['n_episodes']))
    plt.show()

if __name__ == "__main__":
    main()
