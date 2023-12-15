import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import copy
import math


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.phase_len = phase_len  # - number of rounds at each phase
        self.num_arms = num_arms  # - number of content providers
        self.num_users = num_users  # - number of users
        self.arms_thresh = arms_thresh  # - the exposure demands of the content providers
        self.users_distribution = users_distribution  # - the probabilities of the users to arrive

        self.exposure_list_exp = np.zeros(self.num_arms)  # - number of times each arm was chosen in the current phase
        exploration_phase_not_round = (num_arms + num_users) * (
                    pow(num_rounds / (num_arms + num_users), 2 / 3) + pow(math.log(num_rounds),
                                                                          1 / 3))  # # the size of the exploration phase not rounded
        self.exploration_phase = math.ceil(
            exploration_phase_not_round / phase_len) * phase_len  # the size of the exploration phase rounded
        self.current_round = -1  # the number of the current round
        self.user_feedback = np.zeros((num_arms, num_users))  # - the feedback of each user to each arm
        self.user_picks = [[-1] * num_users for _ in range(num_arms)]  # - the number of times each user picked each arm
        self.avg_reward = np.zeros((num_arms, num_users))  # - the average reward of each user from each arm
        self.best_pick = np.zeros(self.num_users)  # - the best arm for each user
        self.phase_counts = np.zeros(self.num_arms)  # - count how many times we need to choose any arm in this phase
        self.last_user = 0  # - the user now
        self.last_arm = 0  # - the arm we picked now
        self.temp_best_pick = np.zeros(self.num_users)  # - the best arm for each user temp
        self.temp_avg_reward = np.zeros((num_arms, num_users))  # - the average reward of each user from each arm temp
        self.removed = []  # the arms we decided to remove

    def exploration(self, user_context):

        if self.current_round % self.phase_len == 0:
            self.phase_counts = copy.deepcopy(self.arms_thresh)

        self.avg_reward = self.user_feedback/self.user_picks
        sum1 = sum(num for num in self.phase_counts if num > 0)
        if sum1 >= self.phase_len - (self.current_round % self.phase_len):
            selected_movies = [movie for movie in range(self.num_arms) if 0 < self.phase_counts[movie]]
        else:
            selected_movies = [movie for movie in range(self.num_arms)]

        ucb_values = []
        for arm in selected_movies:
            x = math.sqrt(2 * math.log(self.exploration_phase))
            if self.user_picks[arm][user_context] == -1:
                user_context_pick = 1
            else:
                user_context_pick = self.user_picks[arm][user_context]
            ucb_value = self.avg_reward[arm][user_context] + (x / user_context_pick)
            ucb_values.append(ucb_value)
        max_ucb_value = max(ucb_values)
        recommended_arm = selected_movies[ucb_values.index(max_ucb_value)]


        if self.user_picks[recommended_arm][user_context] == -1:
            self.user_picks[recommended_arm][user_context] = 0

        self.user_picks[recommended_arm][user_context] += 1
        self.phase_counts[recommended_arm] -= 1
        self.last_arm = recommended_arm
        return recommended_arm

    def info(self):
        self.avg_reward = self.user_feedback / self.user_picks
        changed_arm = False
        while (not changed_arm):
            changed_arm = True
            max_avg_without_i = 0
            max_arm = -1
            max_best_pick = []
            max_temp_avg = []
            isArm = False
            for i in range(self.num_arms):
                if self.num_arms - len(self.removed) <= 1:
                    changed_arm = True
                    break
                temp_avg_reward = copy.deepcopy(self.avg_reward)
                temp_avg_reward[i] = [-3] * self.num_users
                temp_best_pick = np.argmax(self.avg_reward, axis=0)
                best_pick_without_i = np.argmax(temp_avg_reward, axis=0)
                avg_temp_best_pick = 0
                avg_best_pick_without_i = 0
                for j in range(self.num_users):
                    avg_temp_best_pick += self.avg_reward[temp_best_pick[j]][j] * self.users_distribution[j]
                    avg_best_pick_without_i += self.avg_reward[best_pick_without_i[j]][j] * self.users_distribution[j]

                diff_percent = (self.arms_thresh[i] / self.phase_len) + 1
                if avg_temp_best_pick < diff_percent * avg_best_pick_without_i and (i not in self.removed):
                    if diff_percent * avg_best_pick_without_i > max_avg_without_i:
                        max_avg_without_i = diff_percent * avg_best_pick_without_i
                        max_arm = i
                        isArm = True
                        max_temp_avg = temp_avg_reward
                        max_best_pick = best_pick_without_i
            if isArm:
                self.removed.append(max_arm)
                self.best_pick = max_best_pick
                self.avg_reward = max_temp_avg
                changed_arm = False
        self.best_pick = np.argmax(self.avg_reward, axis=0)

    def exploitation(self, user_context):
        if self.current_round % self.phase_len == 0:
            self.temp_avg_reward = copy.deepcopy(self.avg_reward)
            temp_arms = np.zeros(self.num_arms)
            for x in range(self.num_arms):
                temp_arms[x] = self.arms_thresh[x]
            self.phase_counts = temp_arms
            for x in self.removed:
                self.phase_counts[x] = 0

            temp_users = [0] * self.num_users
            for x in range(self.num_users):
                temp_users[x] = self.best_pick[x]
            self.temp_best_pick = temp_users

        sum1 = sum(num for num in self.phase_counts if num > 0)
        if sum1 >= self.phase_len - (self.current_round % self.phase_len):
            for x in range(self.num_arms):
                if self.phase_counts[x] <= 0:
                    self.temp_avg_reward[x] = [-1] * self.num_users
                    self.temp_best_pick = np.argmax(self.temp_avg_reward, axis=0)
        choice = self.temp_best_pick[user_context]
        self.phase_counts[choice] = self.phase_counts[choice] - 1
        return choice

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        self.last_user = user_context
        self.current_round = self.current_round + 1

        if self.current_round < self.exploration_phase:
            return self.exploration(user_context)

        if self.current_round == self.exploration_phase:
            self.info()

        if self.current_round >= self.exploration_phase:
            return self.exploitation(user_context)

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        if self.current_round < self.exploration_phase:
            self.user_feedback[self.last_arm][self.last_user] += reward

    def get_id(self):
        return "id_213761158_211754312"
