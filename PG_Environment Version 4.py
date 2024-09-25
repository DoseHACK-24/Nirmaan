import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from tqdm import tqdm  # Import tqdm for progress bar

class SaveModelCallback(BaseCallback):
    """
    Custom callback for saving a model periodically.
    :param save_freq: Save the model every `save_freq` number of steps.
    :param save_path: Path to save the model.
    """
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"ppo_model_{self.n_calls}_steps.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        return True

class WarehouseEnvWithBotCommands(gym.Env):
    def __init__(self, grid):
        super(WarehouseEnvWithBotCommands, self).__init__()

        self.grid = grid
        self.start_positions = np.argwhere(self.grid == 'A')
        self.goal_positions = np.argwhere(self.grid == 'B')
        self.obstacle_positions = np.argwhere(self.grid == 'X')

        # Define action space (Forward, Reverse, Left turn, Right turn, Wait)
        self.action_space = spaces.Discrete(5)
        
        # Define observation space (the grid itself)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.grid.shape, dtype=np.uint8)

        # Directions: 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.directions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        # Set initial position of the agent and its direction (facing upward)
        self.agent_pos = self.start_positions[0]
        self.agent_dir = 0  # Start facing upward (0 = Up)
        
        # Initialize matplotlib
        self.fig, self.ax = plt.subplots()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_positions[0]
        self.agent_dir = 0  # Reset to facing upward
        return self._get_obs(), {}

    def step(self, action):
        if action == 0:  # Forward
            new_pos = self.agent_pos + np.array(self.directions[self.agent_dir])
            if self._is_valid_move(new_pos):
                self.agent_pos = new_pos

        elif action == 1:  # Reverse
            new_pos = self.agent_pos - np.array(self.directions[self.agent_dir])
            if self._is_valid_move(new_pos):
                self.agent_pos = new_pos

        elif action == 2:  # Turn Left (90 degrees counterclockwise)
            self.agent_dir = (self.agent_dir - 1) % 4  # Rotate left in circular manner

        elif action == 3:  # Turn Right (90 degrees clockwise)
            self.agent_dir = (self.agent_dir + 1) % 4  # Rotate right in circular manner

        elif action == 4:  # Wait (no movement)
            pass

        done = self._is_goal_reached()
        reward = 1 if done else -0.1

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Return the current position of the agent on the grid
        obs = np.zeros_like(self.grid, dtype=np.uint8)
        obs[tuple(self.agent_pos)] = 1  # Mark agent's position
        return obs

    def _is_valid_move(self, pos):
        if (0 <= pos[0] < self.grid.shape[0] and 0 <= pos[1] < self.grid.shape[1]):
            return self.grid[tuple(pos)] != 'X'
        return False

    def _is_goal_reached(self):
        return any(np.array_equal(self.agent_pos, goal) for goal in self.goal_positions)

    def render(self, mode='human'):
        # Set up a color map for the grid
        cmap = colors.ListedColormap(['white', 'black', 'green', 'blue', 'red'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        grid_render = np.zeros_like(self.grid, dtype=np.uint8)

        # Set obstacles (X) as black
        for obs in self.obstacle_positions:
            grid_render[tuple(obs)] = 1

        # Set goals (B) as blue
        for goal in self.goal_positions:
            grid_render[tuple(goal)] = 3

        # Set starting point (A) as green
        grid_render[tuple(self.start_positions[0])] = 2

        # Set the agent (current position) as red
        grid_render[tuple(self.agent_pos)] = 4

        self.ax.imshow(grid_render, cmap=cmap, norm=norm)

        # Update the plot in real time
        plt.draw()
        plt.pause(0.001)


grid = np.array([
    ['A', '.', '.', 'X', 'B'],
    ['.', 'X', '.', '.', '.'],
    ['.', '.', 'X', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['A', 'X', '.', '.', 'B']
])


# Create the environment
env = WarehouseEnvWithBotCommands(grid)

# Create a PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent (without rendering)
total_timesteps = 1000000

# Add the progress bar using tqdm
with tqdm(total=total_timesteps) as pbar:
    obs, _ = env.reset()

    for step in range(total_timesteps):
        # Get the action from the model
        action, _states = model.predict(obs, deterministic=False)
        
        # Apply the action to the environment
        obs, reward, done, truncated, _ = env.step(action)

        # Update the progress bar
        pbar.update(1)

        if done:
            obs, _ = env.reset()

# Close the environment after training
env.close()


# Number of steps to run the agent in the test phase
test_steps = 10

# Initialize variables to track performance
total_reward = 0
episode_length = 0

# Reset the environment to start testing
obs, _ = env.reset()

print("testing")
for step in range(test_steps):
    # Get the action predicted by the model (deterministic for testing)
    action, _states = model.predict(obs, deterministic=True)
    
    # Apply the action and move the bot
    obs, reward, done, truncated, _ = env.step(action)
    
    # Accumulate the reward and count the steps
    total_reward += reward
    
    # Render the bot's movement for visualization
    env.render()

    # If the goal is reached, stop testing
    if done:
        print(f"Goal reached in {step} steps with total reward: {total_reward}")
        break

# If the loop ends without reaching the goal
if not done:
    print(f"Test ended after {step} steps. Total reward: {total_reward}")

# Close the plot after testing
plt.close()
