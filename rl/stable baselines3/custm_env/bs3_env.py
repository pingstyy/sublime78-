from gym import spaces
import gym


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SnakeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
        )

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"): ...
    def close(self): ...
