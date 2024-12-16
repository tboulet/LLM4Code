import gymnasium as gym
import numpy as np
import tensorboardX

# Create the agent
class Agent:
    def __init__(self, action_space):
        self.action_space = action_space  # Continuous action space in Pendulum

    def act(self, observation):
        # Observation: [cos(theta), sin(theta), theta_dot]
        cos_theta, sin_theta, theta_dot = observation

        # Calculate the angle (theta) of the pendulum
        theta = np.arctan2(sin_theta, cos_theta)

        # Control law: Torque proportional to angle and angular velocity
        torque = -1.0 * theta - 0.1 * theta_dot

        # Clip torque to action space bounds
        torque = np.clip(torque, self.action_space.low[0], self.action_space.high[0])
        return np.array([torque])



# Initialize environment and agent
name_env = "Pendulum-v1"
env_train = gym.make(name_env, render_mode=None)
env_test = gym.make(name_env, render_mode="human")
tb_logger = tensorboardX.SummaryWriter("tensorboard/chatGPT_pendulum")

agent = Agent(env_train.action_space)

for ep in range(100):
    # Initialize environment
    print(f"Episode {ep}")
    env = env_train if ep % 50 != 0 else env_test
    observation, info = env.reset()
    terminated = False
    truncated = False
    cum_reward = 0
    
    # Run episode
    while not (terminated or truncated):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Logging
        cum_reward += reward
    
    tb_logger.add_scalar("total_reward", cum_reward, ep)
        
env.close()
