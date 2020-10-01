import numpy as np
import torch
from torch.distributions.uniform import Uniform
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

class Policy():
    """
    General policy framework

    Attributes:
        action_space: Action space of the policy, expected format depends on the
        action_space_type
        representining minimium and maximium values of that action.
        action_space_type: What type of action space the policy is operating in.
        Current possible values are "Discrete" and "Continuous"
        batch_size: Number of actions to be generated at once
    """
    def __init__(self,
                 action_space=None,
                 batch_size=1,
                 device="cpu"):
        self.action_space = action_space
        if isinstance(action_space, Box):
            self.action_range = (action_space.high - action_space.low) / 2
            self.action_range = torch.tensor(self.action_range).to(device)
        self.batch_size = batch_size
        self.device = device
    
    def __call__(self):
        return NotImplementedError

class RandomPolicy(Policy):
    """
    Random Policy

    Attributes:
        action_space: Action space of the policy, expected format depends on the
        action_space_type
        batch_size: Number of actions to be generated at once
        sample_func: Function to use to determine action
    """
    def __init__(self,
                 action_space=None,
                 batch_size=1):
        super().__init__(action_space=action_space,
                         batch_size=batch_size)
        if isinstance(action_space, Discrete):
            self.sample_func = self.sample_discrete
            num_actions = action_space.n
            self.dist = Uniform(torch.tensor([0.0 for _ in range(batch_size)]),
                                torch.tensor([num_actions for _ in range(batch_size)]))
        elif isinstance(action_space, Box):
            self.sample_func = self.sample_continuous
        else:
            raise NotImplementedError


    def sample_discrete(self):
        """
        Samples from a discrete uniform distribution
        """
        action = self.dist.sample().numpy()
        return action.astype(int)
        

    def sample_continuous(self):
        """
        Samples from a continuous uniform distribution

        Returns:
            actions: List of length batch_size, where each element is a list
            that holds the selected value for each dimension.
        """
        # # Generate random values between 0 and 1
        # actions = tf.random.uniform([self.batch_size, self.num_actions])
        # # Convert random actions to appropriate ranges
        # actions = actions * self.ranges + self.mins
        actions = [self.action_space.sample() for _ in range(self.batch_size)]
        return actions

    def __call__(self):
        return self.sample_func()

if __name__ == "__main__":
    from src.general.envs.gym_env import GymEnv
    env = GymEnv("LunarLander-v2")
    obs_space = env.obs_space
    action_space = env.action_space
    policy = RandomPolicy(action_space=action_space, batch_size=2)
    print(policy())

    env = GymEnv("LunarLanderContinuous-v2")
    obs_space = env.obs_space
    action_space = env.action_space
    policy = RandomPolicy(action_space=action_space, batch_size=2)
    print(policy())
