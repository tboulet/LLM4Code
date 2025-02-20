import numpy as np
import random
import time
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=5, max_steps=50):
        # Initialize the grid environment
        self.size = size  # Grid size (size x size)
        self.max_steps = max_steps  # Max steps before termination
        self.idx_to_name_channel = {0: "agent", 1: "goal"}
        self.name_to_idx_channel = {v: k for k, v in self.idx_to_name_channel.items()}
        self.n_channels = len(self.idx_to_name_channel)
        # Initialize rendering
        self.rendering_active = True
        self.reset()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        plt.ion()

    def reset(self):
        """Resets the environment to the initial state."""
        self.grid = np.zeros((self.size, self.size, self.n_channels))  # Initialize grid
        self.grid = np.zeros((self.size, self.size, self.n_channels))  # Initialize grid
        self.agent_pos = np.random.choice(
            [0, self.size - 1], size=2
        )  # Random agent position
        self.goal_pos = np.full(2, self.size // 2)  # Center goal position
        self.grid[self.agent_pos[0], self.agent_pos[1], self.name_to_idx_channel["agent"]] = 1
        self.grid[self.goal_pos[0], self.goal_pos[1], self.name_to_idx_channel["goal"]] = 1
        self.steps = 0
        self.rendering_active = True
        return self.agent_pos

    def step(self, action):
        """Takes a step in the environment."""
        # Update agent position
        idx_agent = self.name_to_idx_channel["agent"]
        self.grid[self.agent_pos[0], self.agent_pos[1], idx_agent] = 0  # Clear previous position
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # Right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        self.grid[self.agent_pos[0], self.agent_pos[1], idx_agent] = 1  # Update agent position
        # Advance step counter
        self.steps += 1
        # Check done
        done = self.steps >= self.max_steps or np.all(self.agent_pos == self.goal_pos)
        return self.agent_pos, done

    def render(self):
        """Renders the grid environment using matplotlib."""
        if not self.rendering_active:
            return None
        self.ax.clear()
        grid_display = np.zeros((self.size, self.size, 3))  # RGB image
        idx_agent = self.name_to_idx_channel["agent"]
        idx_goal = self.name_to_idx_channel["goal"]
        
        grid_display[self.grid[:, :, idx_agent] == 1] = [0, 0, 1]  # Blue for agent
        grid_display[self.grid[:, :, idx_goal] == 1] = [0, 1, 0]  # Green for goal
        
        self.ax.imshow(grid_display, origin='upper')
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        plt.draw()
        plt.pause(0.2)
    
    def on_close(self, event):
        """Stops rendering when the window is closed."""
        self.rendering_active = False

# Run a random agent in the environment
env = GridWorld(size=5, max_steps=50)
state = env.reset()
env.render()
done = False

while not done:
    action = random.randint(0, 3)  # Choose a random action (0-3)
    state, done = env.step(action)
    env.render()

plt.ioff()
plt.close()
