import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import torch
from torch.distributions.normal import Normal


from src.general.policies.policy import Policy

# GLOBAL
# Prevent division by zero
EPS = 1e-6

class SACPolicy(Policy):
    """
    SAC Policy

    Attributes:
        action_space: Action space of the policy, expected format depends on the
        batch_size: Number of actions to be generated at once
        model: Model used for the policy
    """
    def __init__(self,
                 action_space=None,
                 batch_size=1,
                 model=None,
                 device="cpu"):
        super().__init__(action_space=action_space,
                         batch_size=batch_size,
                         device=device)
        assert model
        self.model = model
        if isinstance(action_space, Discrete):
            self.eval_func = self.eval_disc
            self.step_func = self.step_disc
        elif isinstance(action_space, Box):
            self.eval_func = self.eval_cont
            self.step_func = self.step_cont
        else:
            raise NotImplementedError

    def eval_disc(self):
        """
        SAC Discrete coming to cloud engines near you...
        """
        raise NotImplementedError

    def step_disc(self):
        """
        SAC Discrete coming to cloud engines near you...
        """
        raise NotImplementedError

    def eval_cont(self, obs, flag):
        """
        Samples actions using the actor network

        Returns:
            actions: List of length of obs, where each element is a list
            containing actions for all dimensions
        """
        mean, log_std = self.model(obs)
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        pre_squish_action = normal.rsample()
        squish_action = torch.tanh(pre_squish_action)
        action = squish_action * self.action_range

        log_prob = normal.log_prob(pre_squish_action)
        log_prob -= torch.log(self.action_range * (1 - squish_action.pow(2)) + EPS)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob

    def step_cont(self, obs, deterministic=False):
        mean, log_std = self.model(obs)
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        pre_squish_action = mean if deterministic else normal.rsample()
        squish_action = torch.tanh(pre_squish_action)
        action = squish_action * self.action_range

        return action.cpu().detach().numpy()

    def eval(self, obs, flag=False):
        return self.eval_func(obs, flag)

    def step(self, obs, flag=False):
        return self.step_func(obs, flag)

if __name__ == "__main__":
    from src.general.envs.gym_env import GymEnv
    from src.sac.sac_model import SACQNet, SACActor, get_action_space_shape
    env = GymEnv("LunarLanderContinuous-v2")
    obs_space = env.obs_space
    action_space = env.action_space
    action_space_shape = get_action_space_shape(action_space)

    actor = SACActor(state_size=env.obs_space,
                     action_space=env.action_space,
                     fc=(128, 64),
                     conv_size=None)

    policy = SACPolicy(action_space=action_space,
              model=actor)
    
    batch_size = 3
    obs = torch.tensor([obs_space.sample() for _ in range(batch_size)])
    print(obs)
    print(policy.eval(obs))
    print(policy.step(obs))
