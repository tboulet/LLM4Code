import numpy as np
import matplotlib.pyplot as plt
import random

class SimpleMaze:
    def __init__(self, size=(10, 10), dynamic=False):
        """
        Initialize the maze environment.

        :param size: Tuple, the dimensions of the maze (rows, cols).
        :param dynamic: Boolean, if True, walls can change dynamically.
        """
        self.rows, self.cols = size
        self.dynamic = dynamic
        self.reset()

    def reset(self):
        """Resets the maze and places the agent and goal."""
        self.maze = np.zeros((self.rows, self.cols))
        self.t = 0
        
        # Add walls
        for _ in range(self.rows * self.cols // 4):  # Fill approximately 25% with walls
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            self.maze[r, c] = 1  # 1 represents a wall

        # Ensure the agent and goal positions are not walls
        self.agent_pos = (0, 0)
        self.goal_pos = (self.rows - 1, self.cols - 1)
        self.maze[self.agent_pos] = 0
        self.maze[self.goal_pos] = 0

        return self.get_observation(), {}

    def get_observation(self):
        """Returns the current state of the maze."""
        return self.maze, self.agent_pos

    def step(self, action):
        """
        Executes an action and updates the environment.

        :param action: Integer (0=up, 1=right, 2=down, 3=left).
        :return: Tuple (observation, reward, done, info)
        """
        r, c = self.agent_pos
        if action == 0 and r > 0:  # Up
            r -= 1
        elif action == 1 and c < self.cols - 1:  # Right
            c += 1
        elif action == 2 and r < self.rows - 1:  # Down
            r += 1
        elif action == 3 and c > 0:  # Left
            c -= 1

        # Check if the new position is a wall
        if self.maze[r, c] == 1:
            r, c = self.agent_pos  # Reset to the original position

        self.agent_pos = (r, c)

        # Check if the goal is reached
        done = self.agent_pos == self.goal_pos
        reward = 1 if done else -0.01  # Small penalty for each step

        # Optionally, update the maze dynamically
        if self.dynamic:
            self._update_walls()
        
        # Increase the time step and stop after 100 steps
        self.t += 1
        if self.t >= 100:
            done = True
            
        return self.get_observation(), reward, done, False, {}

    def _update_walls(self):
        """Randomly change the walls in the maze."""
        for _ in range(self.rows * self.cols // 20):  # Adjust frequency of wall changes
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (r, c) != self.agent_pos and (r, c) != self.goal_pos:
                self.maze[r, c] = 1 - self.maze[r, c]  # Toggle wall state

    def render(self):
        """Visualize the current state of the maze."""
        plt.ion()
        maze_copy = self.maze.copy()
        maze_copy[self.agent_pos] = 0.5  # Mark agent
        maze_copy[self.goal_pos] = 0.8  # Mark goal

        plt.imshow(maze_copy, cmap="viridis", origin="upper")
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.1)
        plt.draw()
    
    def close(self):
        """Close the visualization."""
        plt.ioff()  # Disable interactive mode
        plt.show()

def struct(container_or_obj):
    if isinstance(container_or_obj, dict):
        return {k: struct(v) for k, v in container_or_obj.items()}
    elif isinstance(container_or_obj, list):
        return [struct(v) for v in container_or_obj]
    # elif isinstance(container_or_obj, tuple):
    #     return tuple([struct(v) for v in container_or_obj])
    else:
        return f"{type(container_or_obj)}"
    
# Example Usage
if __name__ == "__main__":
    env = SimpleMaze(size=(10, 10), dynamic=True)
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = random.randint(0, 3)  # Random action
        print(struct(obs))
        obs, reward, done, trunc, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        input()
    print("Goal reached!")
    
