import pandas as pd
import intraday_environment
from sac_tf2 import Agent
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt

n_episodes = 10000

# Environment
SOC = 50
df = pd.read_csv('test_from_11_0to_13_45.csv')

environment = intraday_environment.IntradayEnv(df, SOC)
env = tf_py_environment.TFPyEnvironment(environment)

# Agent parameters
n_actions = env.action_spec().shape[0]
input_dims = env.observation_spec().shape
max_action = env.action_spec().maximum.max()

# Initialize agent
agent = Agent(n_actions, input_dims, max_action, batch_size=64)

# Training
score_history = []
value_loss = []
actor_loss = []
critic1_loss = []
critic2_loss = []

plt.ion()
# fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

for i in range(n_episodes):

    observation = environment.reset_new()
    done = False
    score = 0

    # Start the episode and run until it's finished
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = environment.step_new(action)
        score = score + environment.total_revenue

        agent.remember(observation, action, reward, observation_, done)

        agent.learn()

        observation = observation_

    # Keep track of the episode's score
    score_history.append(score)

    # Print training info
    print(f'episode: {i}, score: {score:.2f}')

    if len(agent.value_loss_log):
        value_loss.append(sum(agent.value_loss_log[-environment.episode_steps:])/environment.episode_steps)
        actor_loss.append(sum(agent.actor_loss_log[-environment.episode_steps:]) / environment.episode_steps)
        critic1_loss.append(sum(agent.critic1_loss_log[-environment.episode_steps:]) / environment.episode_steps)
        critic2_loss.append(sum(agent.critic2_loss_log[-environment.episode_steps:]) / environment.episode_steps)

        # ax1.plot(value_loss)
        # ax2.plot(actor_loss)
        # ax3.plot(critic1_loss)
        # ax4.plot(critic2_loss)
        # plt.draw()
        # plt.pause(0.02)

        print(value_loss)
        print(actor_loss)
        print(critic1_loss)
        print(critic2_loss)

    print(score_history)

# Save the model
agent.save_models()

# Plot the score curve
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.plot(value_loss)
ax2.plot(actor_loss)
ax3.plot(critic1_loss)
ax4.plot(critic2_loss)
plt.draw()

x = [i for i in range(n_episodes)]
fig = plt.figure()
plt.plot(x, score_history)
plt.title('Score over episodes')
plt.ioff()
plt.show()
