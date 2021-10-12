import auxilary_functions as aux
import lob
import numpy as np
from datetime import datetime, timedelta
import copy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from sklearn.preprocessing import MinMaxScaler

step_count = 0

# Product
N_PRODUCTS = 12

# Quantiles
N_PERC = 3  # Total number of percentiles for each state variable
START_PERC = 5  # Starting percentile
END_PERC = 60  # Starting percentile

# Action space variables
ACTION_SPACE_SIZE = N_PRODUCTS
MAX_BUY_POWER = 1.0
MAX_SELL_POWER = -1.0

# State space variables
STATE_SPACE_SIZE = (4 * N_PERC + 2) * N_PRODUCTS
MIN_V = 0.0
MAX_V = 10000.0
MIN_P = -3000.0
MAX_P = 3000.0
MIN_SOC = 0.0
MAX_SOC = 100.0
MIN_RT = 0
MAX_RT = 4

i_vol_start = 0  # including
i_vol_end = N_PRODUCTS * 2 * N_PERC  # excluding
i_price_start = N_PRODUCTS * 2 * N_PERC  # including
i_price_end = N_PRODUCTS * 4 * N_PERC  # excluding
i_soc_start = N_PRODUCTS * 4 * N_PERC  # including
i_soc_end = N_PRODUCTS * 4 * N_PERC + N_PRODUCTS  # excluding
i_rt_start = N_PRODUCTS * 4 * N_PERC + N_PRODUCTS  # including
i_rt_end = N_PRODUCTS * 4 * N_PERC + 2 * N_PRODUCTS  # excluding

# Reward variables
MIN_REWARD = -1000
GAMMA = 1.0

# SAC
max_action = 15

# Application variables
PRODUCT_DUR = 0.25  # product duration in h, for example 15min = 0.25h

MAX_CAPACITY = 200  # in MWh

MIN_SOC_CAPACITY = 30  # in %
MAX_SOC_CAPACITY = 70  # in %

MIN_SOC_CAPACITY_END = 48  # in %
MAX_SOC_CAPACITY_END = 52  # in %

DESIRED_DATE = '06/12/2019'
t_start = datetime(2019, 12, 6, 9, 00, 0)
t_end = datetime(2019, 12, 6, 13, 00, 0)
dt = 5 * 60  # in seconds
closing_dt = 1  # time before delivery that market closes (in hours)


class IntradayEnv(py_environment.PyEnvironment):

    def __init__(self, df, soc):
        self._action_spec = array_spec.BoundedArraySpec(shape=(ACTION_SPACE_SIZE,),
                                                        dtype=np.float32,
                                                        minimum=MAX_SELL_POWER,
                                                        maximum=MAX_BUY_POWER,
                                                        name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(STATE_SPACE_SIZE,),
        #                                                      dtype=np.float32,
        #                                                      minimum=N_PRODUCTS * (2 * (N_PERC * [MIN_V] + N_PERC * [MIN_P]) + [MIN_SOC]),
        #                                                      maximum=N_PRODUCTS * (2 * (N_PERC * [MAX_V] + N_PERC * [MAX_P]) + [MAX_SOC]),
        #                                                      name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(STATE_SPACE_SIZE,),
                                                             dtype=np.float32,
                                                             minimum=N_PRODUCTS * 2 * N_PERC * ([MIN_V] + [MIN_P]) + (N_PRODUCTS * [MIN_SOC]) + (N_PRODUCTS * [MIN_RT]),
                                                             maximum=N_PRODUCTS * 2 * N_PERC * ([MAX_V] + [MAX_P]) + (N_PRODUCTS * [MAX_SOC]) + (N_PRODUCTS * [MAX_RT]),
                                                             name='observation')

        # Create a limit order book object
        self.lb = lob.LOB(DESIRED_DATE, df)
        self.lb.reconstructTillTime(t_start)

        # Get the expected delivery times
        self.delivery_time = np.sort(self.lb.products)

        # Set the current time
        self.current_time = t_start

        # Store the initial values at the start of the episode; used for the reset as well
        self.initial_buy_list = copy.deepcopy(self.lb.buy_list)
        self.initial_sell_list = copy.deepcopy(self.lb.sell_list)
        self.initial_soc = soc

        self.current_soc = soc

        # Expected state of charge at each delivery time
        self.soc = self.initial_soc * np.ones((N_PRODUCTS,), dtype=np.float32)

        # Get the available products including in the lob
        self.product_ids = np.sort(list(self.lb.buy_list.keys()))

        # Set the normalization objects
        self.scaler_v = MinMaxScaler()
        self.scaler_p = MinMaxScaler()

        self.episode_steps = 0

        self.total_revenue = 0

        self.episode_ended = False

        self.state = np.zeros(shape=(STATE_SPACE_SIZE,), dtype=np.float32)  # invoke a function to set state to the corresponding values
        self.setState_new()  # assign the initial values to the state vector

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.lb.clear_object()
        # self.lb.reconstructTillTime(t_start)

        self.lb.buy_list = copy.deepcopy(self.initial_buy_list)
        self.lb.sell_list = copy.deepcopy(self.initial_sell_list)
        self.soc = self.initial_soc * np.ones((N_PRODUCTS,), dtype=np.float32)

        self.current_soc = self.initial_soc

        self.current_time = t_start
        # self.step_reward = 0

        self.episode_steps = 0

        self.total_revenue = 0

        self.setState()  # assign the initial values to the state vector

        self.episode_ended = False

        return ts.restart(self.state)  # returns a TimeStep object with a step_type=StepType.FIRST and observation=state

    def _step(self, action):

        if self.episode_ended:
            return self.reset()

        # For debugging
        global step_count
        step_count = step_count + 1
        if step_count % 200 == 0:
            print(f'Step count: {step_count}')
        # end

        self.episode_steps = self.episode_steps + 1
        print(f'Current step: {self.episode_steps}, current time: {self.current_time}')

        step_reward = 0
        traded = N_PRODUCTS * [False]

        for i, product_id in enumerate(self.product_ids):

            if self.current_time < (self.delivery_time[i] - timedelta(hours=closing_dt)):

                if action[i] > 0:
                    traded[i], action_revenue = self.updateSellList(action[i], product_id)  # I buy from the sell_list
                    action_reward = -action_revenue
                else:
                    traded[i], action_revenue = self.updateBuyList(action[i], product_id)  # I sell to the buy_list
                    action_reward = 2 * action_revenue

                step_reward = step_reward + action_reward

                self.total_revenue = self.total_revenue + action_revenue

                if traded[i]:
                    energy_traded = action[i] * PRODUCT_DUR
                else:
                    energy_traded = 0

                if i == 0:
                    self.soc[i] = self.current_soc + 100 * (energy_traded / MAX_CAPACITY)
                    self.current_soc = self.soc[i]
                else:
                    self.soc[i] = self.soc[i - 1] + 100 * (energy_traded / MAX_CAPACITY)

                print(
                    f'\t Action: {action[i]:.2f}, Action revenue: {action_revenue:.2f}, SoC: {self.soc[i]:.2f}, Traded: {traded[i]}')
                # print(f'\t Action: {action[i]:.2f}')

            # If the product has been consumed
            else:
                # action[i] = 0
                traded[i] = True  # it wasn't actually traded, but we want the episode to continue running
                self.lb.buy_list[product_id] = []
                self.lb.sell_list[product_id] = []

        self.setState()

        # Check if episode is ended
        if self.current_time > (self.delivery_time[-1] - timedelta(hours=closing_dt)) \
                or self.current_time > t_end:

            self.episode_ended = True
            return ts.termination(self.state, reward=0)

        # Check if this soc is feasible or I bought/sold more than existed in the market
        elif any(soc < MIN_SOC_CAPACITY or soc > MAX_SOC_CAPACITY for soc in self.soc) \
                or self.soc[-1] < MIN_SOC_CAPACITY_END or self.soc[-1] > MAX_SOC_CAPACITY_END \
                or not all(traded_i for traded_i in traded):

            reward = MIN_REWARD

            # self.episode_ended = True
            #
            # print(f'Total number of steps at this episode: {self.episode_steps}')
            #
            # return ts.termination(self.state, reward)

            self.current_time = self.current_time + timedelta(seconds=dt)
            self.lb.reconstructTillTime(self.current_time)

            self.setState()

            print(f'Reward: {reward}')
            return ts.transition(self.state, reward=reward, discount=GAMMA)

        # If trades were normally performed without any violation
        else:
            reward = step_reward

            self.current_time = self.current_time + timedelta(seconds=dt)
            self.lb.reconstructTillTime(self.current_time)

            self.setState()

            print(f'Reward: {reward}')
            return ts.transition(self.state, reward=reward, discount=GAMMA)

    def setState(self):
        state_ = []

        for i, product_id in enumerate(self.product_ids):
            # Add buy info
            if len(self.lb.buy_list[product_id]) == 0:
                state_ = state_ + N_PERC * [0]  # volumes
                state_ = state_ + N_PERC * [0]  # prices
            else:
                state_ = state_ + aux.calcVolumePercentiles(self.lb.buy_list[product_id], N_PERC, START_PERC)[::-1]  # volumes
                state_ = state_ + aux.calcPricePercentiles(self.lb.buy_list[product_id], N_PERC, START_PERC)[::-1]  # prices

            # Add sell info
            if len(self.lb.sell_list[product_id]) == 0:
                state_ = state_ + N_PERC * [0]  # volumes
                state_ = state_ + N_PERC * [0]  # prices
            else:
                state_ = state_ + aux.calcVolumePercentiles(self.lb.sell_list[product_id], N_PERC, START_PERC)  # volumes
                state_ = state_ + aux.calcPricePercentiles(self.lb.sell_list[product_id], N_PERC, START_PERC)  # prices

            # Add SoC info
            state_ = state_ + [self.soc[i]]

        self.state = np.array(state_, dtype=np.float32)

        # print(self.state)

    def updateBuyList(self, action, product_id):

        action = abs(action)  # sell action is negative

        action_revenue = 0

        max_available_volume = np.sum(aux.returnVolume(self.lb.buy_list[product_id]))

        # print(f'\t Buy at {product_id}, Total amount: {max_available_volume:.2f}, requested: {action:.2f}')

        if max_available_volume < action:
            # self.lb.buy_list[product_id] = []

            return False, action_revenue
        else:
            self.lb.buy_list[product_id] = sorted(self.lb.buy_list[product_id], key=lambda x: x.price, reverse=True)

            remaining_volume_needed = action

            while remaining_volume_needed > 0:

                # Since buy_list is sorted, I will always take the first index (=0)
                if self.lb.buy_list[product_id][0].volume <= remaining_volume_needed:

                    action_revenue = action_revenue + (self.lb.buy_list[product_id][0].volume * PRODUCT_DUR * self.lb.buy_list[product_id][0].price)

                    remaining_volume_needed = remaining_volume_needed - self.lb.buy_list[product_id][0].volume
                    self.lb.buy_list[product_id].pop(0)
                else:

                    action_revenue = action_revenue + (remaining_volume_needed * PRODUCT_DUR * self.lb.buy_list[product_id][0].price)

                    self.lb.buy_list[product_id][0].volume = self.lb.buy_list[product_id][0].volume - remaining_volume_needed
                    remaining_volume_needed = 0

            return True, action_revenue

    def updateSellList(self, action, product_id):

        action_revenue = 0

        max_available_volume = np.sum(aux.returnVolume(self.lb.sell_list[product_id]))

        # print(f'\t Sell at {product_id}, Total amount: {max_available_volume:.2f}, requested: {action:.2f}')

        if max_available_volume < action:
            # self.lb.sell_list[product_id] = []

            return False, action_revenue
        else:
            self.lb.sell_list[product_id] = sorted(self.lb.sell_list[product_id], key=lambda x: x.price, reverse=False)

            remaining_volume_needed = action

            while remaining_volume_needed > 0:

                # Since sell_list is sorted, I will always take the first index (=0)
                if self.lb.sell_list[product_id][0].volume <= remaining_volume_needed:

                    action_revenue = action_revenue - (self.lb.sell_list[product_id][0].volume * PRODUCT_DUR * self.lb.sell_list[product_id][0].price)

                    remaining_volume_needed = remaining_volume_needed - self.lb.sell_list[product_id][0].volume
                    self.lb.sell_list[product_id].pop(0)
                else:

                    action_revenue = action_revenue - (remaining_volume_needed * PRODUCT_DUR * self.lb.sell_list[product_id][0].price)

                    self.lb.sell_list[product_id][0].volume = self.lb.sell_list[product_id][0].volume - remaining_volume_needed
                    remaining_volume_needed = 0

            return True, action_revenue

    def step_new(self, action):

        info = {}  # TODO: diagnostic information for debugging

        action = action.numpy()  # convert Tensor to numpy array
        action = action * max_action

        # For debugging
        # global step_count
        # step_count = step_count + 1
        # if step_count % 200 == 0:
        #     print(f'Step count: {step_count}')
        # end

        self.episode_steps = self.episode_steps + 1
        print(f'Current step: {self.episode_steps}, current time: {self.current_time}')

        step_reward = 0
        traded = N_PRODUCTS * [False]

        for i, product_id in enumerate(self.product_ids):

            if self.current_time < (self.delivery_time[i] - timedelta(hours=closing_dt)):

                if action[i] > 0:
                    traded[i], action_revenue = self.updateSellList(action[i], product_id)  # I buy from the sell_list
                    action_reward = -action_revenue
                else:
                    traded[i], action_revenue = self.updateBuyList(action[i], product_id)  # I sell to the buy_list
                    action_reward = 2 * action_revenue

                step_reward = step_reward + action_reward

                self.total_revenue = self.total_revenue + action_revenue

                if traded[i]:
                    energy_traded = action[i] * PRODUCT_DUR
                else:
                    energy_traded = 0

                if i == 0:
                    self.soc[i] = self.current_soc + 100 * (energy_traded / MAX_CAPACITY)
                    self.current_soc = self.soc[i]
                else:
                    self.soc[i] = self.soc[i - 1] + 100 * (energy_traded / MAX_CAPACITY)

                print(
                    f'\t Action: {action[i]:.2f}, Action revenue: {action_revenue:.2f}, SoC: {self.soc[i]:.2f}, Traded: {traded[i]}')
                # print(f'\t Action: {action[i]:.2f}')

            # If the product has been consumed
            else:
                # action[i] = 0
                traded[i] = True  # it wasn't actually traded, but we want the episode to continue running
                self.lb.buy_list[product_id] = []
                self.lb.sell_list[product_id] = []

        # self.setState_new()

        # Check if episode has ended
        if self.current_time > (self.delivery_time[-1] - timedelta(hours=closing_dt)) \
                or self.current_time > t_end:

            reward = 0
            self.episode_ended = True
            self.setState_new()

        # Check if this soc is feasible or I bought/sold more than existed in the market
        elif any(soc < MIN_SOC_CAPACITY or soc > MAX_SOC_CAPACITY for soc in self.soc) \
                or self.soc[-1] < MIN_SOC_CAPACITY_END or self.soc[-1] > MAX_SOC_CAPACITY_END \
                or not all(traded_i for traded_i in traded):

            reward = MIN_REWARD

            self.total_revenue = 0

            self.current_time = self.current_time + timedelta(seconds=dt)
            self.lb.reconstructTillTime(self.current_time)

            self.setState_new()

            self.episode_ended = True

        # If trades were normally performed without any violation
        else:
            reward = step_reward

            self.current_time = self.current_time + timedelta(seconds=dt)
            self.lb.reconstructTillTime(self.current_time)

            self.setState_new()

        print(f'Reward: {reward}')

        return self.state, reward, self.episode_ended, info

    def reset_new(self):
        self.lb.clear_object()
        # self.lb.reconstructTillTime(t_start)

        self.lb.buy_list = copy.deepcopy(self.initial_buy_list)
        self.lb.sell_list = copy.deepcopy(self.initial_sell_list)
        self.soc = self.initial_soc * np.ones((N_PRODUCTS,), dtype=np.float32)

        self.current_soc = self.initial_soc

        self.current_time = t_start
        # self.step_reward = 0

        self.episode_steps = 0

        self.total_revenue = 0

        self.setState_new()  # assign the initial values to the state vector

        self.episode_ended = False

        return self.state

    def setState_new(self):
        state_ = []

        # Add volume info
        for i, product_id in enumerate(self.product_ids):
            if len(self.lb.buy_list[product_id]) == 0:
                state_ = state_ + N_PERC * [0]  # volumes
            else:
                state_ = state_ + aux.calcVolumePercentiles(self.lb.buy_list[product_id], N_PERC, START_PERC)[::-1]  # volumes
            if len(self.lb.sell_list[product_id]) == 0:
                state_ = state_ + N_PERC * [0]  # volumes
            else:
                state_ = state_ + aux.calcVolumePercentiles(self.lb.sell_list[product_id], N_PERC, START_PERC)  # volumes

        # Add price info
        for i, product_id in enumerate(self.product_ids):
            if len(self.lb.buy_list[product_id]) == 0:
                state_ = state_ + N_PERC * [0]  # prices
            else:
                state_ = state_ + aux.calcPricePercentiles(self.lb.buy_list[product_id], N_PERC, START_PERC)[::-1]  # prices
            if len(self.lb.sell_list[product_id]) == 0:
                state_ = state_ + N_PERC * [0]  # prices
            else:
                state_ = state_ + aux.calcPricePercentiles(self.lb.sell_list[product_id], N_PERC, START_PERC)  # prices

        # Add SoC info
        state_ = state_ + [soc_i for soc_i in self.soc]

        # Add remaining time info
        remaining_time = self.delivery_time - self.current_time
        state_ = state_ + [rt.seconds / 3600 for rt in remaining_time]

        self.state = np.array(state_, dtype=np.float32)

        if self.episode_steps == 0:
            self.scaler_v.fit(self.state[i_vol_start:i_vol_end].reshape(-1, 1))
            self.scaler_p.fit(self.state[i_price_start:i_price_end].reshape(-1, 1))

        self.normalizeStates()

    def normalizeStates(self):
        self.state[i_vol_start:i_vol_end] = np.squeeze(self.scaler_v.transform(self.state[i_vol_start:i_vol_end].reshape(-1, 1)))
        self.state[i_price_start:i_price_end] = np.squeeze(self.scaler_p.transform(self.state[i_price_start:i_price_end].reshape(-1, 1)))
        self.state[i_soc_start:i_soc_end] = (self.state[i_soc_start:i_soc_end] - MIN_SOC) / (MAX_SOC - MIN_SOC)
        self.state[i_rt_start:i_rt_end] = (self.state[i_rt_start:i_rt_end] - MIN_RT) / (MAX_RT - MIN_RT)
