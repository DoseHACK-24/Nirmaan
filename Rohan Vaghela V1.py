import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class WarehouseEnv(gym.Env):
    """Custom Environment with graphical visualization using Matplotlib."""
    metadata = {'render.modes': ['human']}
    
    def _init_(self, rows, cols, obstacles, start_pos, goal_pos):
        super(WarehouseEnv, self)._init_()
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))  # Empty grid initialized
        self.obstacles = obstacles
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos

        # Define action and observation space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=max(self.rows, self.cols), shape=(2,), dtype=np.int32)

        # Mark obstacles on the grid
        for (x, y) in obstacles:
            self.grid[x][y] = -1  # Obstacle marked as -1

        # Create a figure and axis for rendering
        self.fig, self.ax = plt.subplots()

    def reset(self):
        """ Reset the environment to the initial state """
        self.current_pos = self.start_pos
        return np.array(self.current_pos)

    def step(self, action):
        """ Execute one time step in the environment """
        x, y = self.current_pos
        
        if action == 0:  # Forward
            new_pos = (x-1, y)
        elif action == 1:  # Reverse
            new_pos = (x+1, y)
        elif action == 2:  # Left
            new_pos = (x, y-1)
        elif action == 3:  # Right
            new_pos = (x, y+1)
        elif action == 4:  # Wait
            new_pos = (x, y)  # No movement
        
        # Ensure the move is valid
        if self.is_valid_position(new_pos):
            self.current_pos = new_pos
        
        # Calculate reward
        reward = self.get_reward()
        
        # Check if episode is done
        done = bool(self.current_pos == self.goal_pos)
        
        return np.array(self.current_pos), reward, done, {}

    def is_valid_position(self, pos):
        """ Check if a position is within the grid and not an obstacle """
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] == 0

    def get_reward(self):
        """ Get the reward for the current position """
        if self.current_pos == self.goal_pos:
            return 100  # Reached destination
        elif self.current_pos in self.obstacles:
            return -10  # Hit an obstacle
        else:
            return -1  # Time penalty

    def render(self, mode='human'):
        """ Render the environment using Matplotlib """
        self.ax.clear()  # Clear previous render

        # Draw grid
        for x in range(self.rows):
            for y in range(self.cols):
                self.ax.add_patch(Rectangle((y, x), 1, 1, fill=False, edgecolor='gray'))

        # Draw obstacles
        for (x, y) in self.obstacles:
            self.ax.add_patch(Rectangle((y, x), 1, 1, color='black'))

        # Draw goal
        goal_x, goal_y = self.goal_pos
        self.ax.add_patch(Rectangle((goal_y, goal_x), 1, 1, color='green'))

        # Draw bot
        bot_x, bot_y = self.current_pos
        self.ax.add_patch(Rectangle((bot_y, bot_x), 1, 1, color='blue'))

        # Set the axis limits and show the grid
        self.ax.set_xlim([0, self.cols])
        self.ax.set_ylim([0, self.rows])
        self.ax.set_xticks(np.arange(0, self.cols, 1))
        self.ax.set_yticks(np.arange(0, self.rows, 1))
        self.ax.grid(True)

        # Update the plot
        plt.draw()
        plt.pause(0.1)  # Small pause for visualization

    def close(self):
        plt.close()

from stable_baselines3 import PPO

# Initialize the environment
obstacles = [(1, 1), (2, 2), (3, 3)]
env = WarehouseEnv(rows=5, cols=5, obstacles=obstacles, start_pos=(0, 0), goal_pos=(4, 4))

# Train using PPO algorithm
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

env.close()  # Close the visualization window


# Load the trained model
model = PPO.load("ppo_warehouse_agent")

# Reset the environment
obs = env.reset()

for i in range(100):  # Run for 100 steps or until done
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # Visualize each step
    if done:
        print("Goal reached!")
        break