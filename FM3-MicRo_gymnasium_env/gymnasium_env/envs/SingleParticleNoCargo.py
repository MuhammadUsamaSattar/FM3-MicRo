import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gymnasium_env.envs.system.Library import functions
from gymnasium_env.envs.system.simulator import Simulator
from gymnasium_env.envs.system import initializations


class SingleParticleNoCargo(gym.Env):
    """Custom Gymnasium Environment wrapping a Pygame-based Simulator."""
    metadata = {'render_modes': ['human'], "framerate" : 60}

    def __init__(self, render_mode=None):
        # Initialize the simulator
        print(render_mode)
        self.simulator = Simulator(self.metadata['framerate'])

        # Define observation space: 2D particle location (-r to r) and 2D goal location (-r to r)
        r = initializations.SIM_SOL_CIRCLE_RAD
        self.observation_space = spaces.Dict({
            'particle_loc': spaces.Box(low=-r, high=r, shape=(2,), dtype=np.float32),
            'goal_loc': spaces.Box(low=-r, high=r, shape=(2,), dtype=np.float32),
        })

        # Define action space: 8 coil currents, each between 0 and 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        self.running = True
        self.prev_time = time.time()

        assert render_mode == None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        particle_loc, goal_loc, _, _ = self.simulator.getState()

        return {
            "particle_loc": np.array(particle_loc, dtype=np.float32),
            "goal_loc": np.array(goal_loc, dtype=np.float32),
        }

    def _get_info(self):
        particle_loc, goal_loc, _, _ = self.simulator.getState()

        return {
            "distance": functions.distance(
                particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
            )
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.simulator.resetAtRandomLocs(seed)
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Apply action and return the new observation, reward, done, and info."""
        render = False
        if str(self.render_mode) == 'human':
            render = True

        # Run one frame of the simulator
        self.running, self.prev_time = self.simulator.step(self.running, self.prev_time, action, render)

        # Get the current state from the simulator
        particle_loc, goal_loc, coil_vals, _ = self.simulator.getState()

        # Extract necessary information from the state
        obs = self._get_obs()
        info = self._get_info()

        # Calculate the reward as the negative distance to the goal
        reward = -info["distance"]

        # Determine if the episode is done (if the particle is close to the goal)
        done = info["distance"] < initializations.SIM_MULTIPLE_GOALS_ACCURACY or not self.running  # Done if close to goal or simulator is closed

        # Prepare the observation
        obs = self._get_obs()

        self.close()

        return obs, reward, done, False, info

    def render(self):
        """Render the environment (for now, just use the simulator's rendering)."""
        # You could add a visualization here or just rely on the Pygame window
        pass

    def close(self):
        """Close the environment and simulator."""
        if self.running == False:
            self.simulator.close()
            sys.exit()