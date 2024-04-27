import numpy as np
import torch
from buffer import ReplayBuffer
from utils import collect_random

from collections import Counter




class Multi_Agent:
    def __init__(self, agent_class, env, config, agent_id):
        self.config = config
        self.env = env
        self.agent_id= agent_id
        self.len_episode = int(self.config['DRL']['Max_Episode'])
        self.agents = agent_class(self.env, config, self.agent_id)
        self.drl_class = self.config['DRL']['_class_ML']

    def run(self):
        score_log = []
        diffpro_log = []
        battery_log = []
        pridction_PV = []
        prediction_Consum = []
        list_r=[]
        done = False
        score = 0
        state = self.env.reset()
        state_log = np.array([])
        action_log = np.array([])
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])
        steps = 0
        steps_ = 0
        eps = 1.
        d_eps = 1 - 0.01 #Minimal Epsilon
        if self.drl_class == 'DSAC':
            state = self.env.reset()
            load_data, load_data_time = self.env.load_data()
            '''
            np.random.seed(1)
            random.seed(1)
            torch.manual_seed(1)
            self.env.seed(1)
            self.env.action_space.seed(1)
            '''
            buffer = ReplayBuffer(buffer_size=100_000, batch_size=256,
                              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            collect_random(env=self.env, dataset=buffer, num_samples=1000)
        





        while not done:
            # get action for the current state and go one step in environment
            state_ = state.copy()
            state_ = np.transpose(state_)
            if steps == 0:
                state_log = np.hstack((state_log, state_))
            else:
                state_log = np.vstack((state_log, state_))
            if self.drl_class == 'DSAC':
                action = self.agents.get_action(state)
                steps += 1
                #action = randrange(2)

                next_state, reward, done, info = self.env.step(action)
                buffer.add(state, action, reward, next_state, done)
                self.agents.learn(steps, buffer.sample(), gamma=0.99)
                state = next_state
                score += reward
                list_r.append(reward)
                



            pridction_PV.append(abs(action[1] - state[-2]))
            prediction_Consum.append(abs(action[2] - state[-1]))
            if steps == 1:
                action_log = np.hstack((action_log, action))
            else:
                action_log = np.vstack((action_log, action))
            diffpro_log.append(info.get('dic_diffpro'))
            battery_log.append(info.get('dic_battery'))

            print("Emergency Situation is {}; Emergy response is {}; prediction_response is {}".format(state[-2], action[1], pridction_PV[-1]))




            if done:

                self.agents.avg_reward_per_house.append(np.mean(list_r))
                print("step------------------------------", steps)
              

                self.agents.Emergency_response_PV = Counter([a - b for a, b in zip(state_log[:,-2], action_log[:, 1])])[0.0] / len(state_log) * 100
                self.agents.Emergency_response_Consum = Counter([a - b for a, b in zip(state_log[:,-1], action_log[:, 2])])[0.0] / len(state_log) * 100
                self.agents.best_score = score
                #self.agents.best_score = np.mean(list_r)
                self.agents.best_diffpro = np.mean(diffpro_log)
                self.agents.best_battery = np.mean(battery_log)
                print("Best score is {}; Average reward is {}; ".format(self.agents.best_score, np.mean(list_r)))


                return load_data, load_data_time, self.agents.best_score, self.agents.avg_reward_per_house, pridction_PV, prediction_Consum, action_log, state_log, diffpro_log, battery_log, self.agents.Emergency_response_PV, self.agents.Emergency_response_Consum, self.agents.best_score, self.agents.best_diffpro, self.agents.best_battery
