import numpy as np
import torch
import torch.nn as nn
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from src.general.models.model import Model

def get_action_space_shape(action_space):
    if isinstance(action_space, Box):
        return action_space.shape
    else:
        raise NotImplementedError

class SACQNet(Model):
    """
    Soft Actor Critic Q Model

    Attributes:
        state_size: State size the model will accept
        convs: Convolutional layers of the model
        flatten: Flatten operation of the model
        fc: List containing the model's fully connected layers
        out: Output layer of the model
        activation: Model activation operation
    """
    def __init__(self,
                 state_size=None,
                 action_space=None,
                 fc=None,
                 conv_size=None):
        """
        Constructor.

        Parameters:
            state_size: List containing the expected size of the state
            action_space: Box action space the actor will work in
            fc: Iterable containing the amount of neurons per layer
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            conv_size: Iterable containing the kernel size, stride and number
            of filters for each convolutional layer.
                ex: ((8, 4, 16)) would make 1 convolution layer with an 8x8
                kernel,(4, 4) stride and 16 filters
        """
        super().__init__()

        # Build convolutional layers
        self.convs = make_convs(conv_size, state_size)
        self.flatten = nn.Flatten(start_dim=1)
        action_space_shape = get_action_space_shape(action_space)
        if len(action_space_shape) > 1:
            raise NotImplementedError

        # Find size of input to dense layers
        if self.convs:
            _in = self.convs(np.ones(state_size.shape, dtype=np.float32)[None, :],
                             np.ones(action_space_shape, dtype=np.float32)[None, :])
            _in = self.flatten(_in)
            prev_linear = _in.shape[1]
        else:
            assert len(state_size.shape) == 1
            prev_linear = state_size.shape[0]
        prev_linear += action_space_shape[0]

        # Build the dense layers
        self.fc = nn.ModuleList()
        for neurons in fc:
            self.fc.append(nn.Linear(prev_linear, neurons))
            prev_linear = neurons
        self.out = nn.Linear(prev_linear, 1)

        self.activation = nn.ReLU()

    def forward(self, obs, actions):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        dense = torch.cat([dense_in, actions], axis=1)
        for l in self.fc:
            dense = l(dense)
            dense = self.activation(dense)
        out = self.out(dense)

        return out

class SACActor(Model):
    """
    Soft Actor Critic Actor Model

    Attributes:
        state_size: State size the model will accept
        convs: Convolutional layers of the model
        flatten: Flatten operation of the model
        fc: List containing the model's fully connected layers
        mean, std: Output layers of the model
        activation: Model activation operation
    """
    def __init__(self,
                 state_size=None,
                 action_space=None,
                 fc=None,
                 conv_size=None):
        """
        Constructor.

        Parameters:
            state_size: List containing the expected size of the state
            action_space: Box action space the actor will work in
            fc: Iterable containing the amount of neurons per layer
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            conv_size: Iterable containing the kernel size, stride and number
            of filters for each convolutional layer.
                ex: ((8, 4, 16)) would make 1 convolution layer with an 8x8
                kernel,(4, 4) stride and 16 filters
        """
        super().__init__()

        # Build convolutional layers
        self.convs = make_convs(conv_size, state_size)
        self.flatten = nn.Flatten(start_dim=1)
        action_space_shape = get_action_space_shape(action_space)
        if len(action_space_shape) > 1:
            raise NotImplementedError

        # Get true input_size
        self.num_actions = get_action_space_shape(action_space)[0]

        # Find size of input to dense layers
        if self.convs:
            _in = self.convs(np.ones(state_size.shape, dtype=np.float32)[None, :])
            _in = self.flatten(_in)
            prev_linear = _in.shape[1]
        else:
            assert len(state_size.shape) == 1
            prev_linear = state_size.shape[0]

        # Build the dense layers
        self.fc = nn.ModuleList()
        for neurons in fc:
            self.fc.append(nn.Linear(prev_linear, neurons))
            prev_linear = neurons
        self.mean = nn.Linear(prev_linear, self.num_actions)
        self.std = nn.Linear(prev_linear, self.num_actions)

        self.activation = nn.ReLU()

    def forward(self, obs):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        # Run actor layers
        dense = dense_in
        for l in self.fc:
            dense = l(dense)
            dense = self.activation(dense)
        mean = self.mean(dense)
        std = self.std(dense)

        return mean, std

def make_convs(conv_size, state_size):
    """
    Build convolutions based on passed parameter
    """
    if conv_size is not None:
        if isinstance(conv_size, tuple):
            return Custom_Convs(conv_size, state_size)
        elif conv_size == "quake":
            return Quake_Block(state_size)
        else:
            raise ValueError("Invalid CNN Topology")
    else:
        return None

class Custom_Convs(nn.Module):
    """
    Custom Convolution Block
    """
    def __init__(self, conv_size, stat_size):
        super().__init__(name='')

        self.activation = nn.ReLU()
        self.convs = []
        prev_channel = state_size[-1]
        for i,(k,s,f) in enumerate(conv_size):
            conv = nn.Conv2d(in_channels=prev_channel,
                      out_channels=filters,
                      kernel_size=k,
                      stride=s,
                      activation=actv)
            self.convs.append(conv)
            prev_channel = filters
    
    def call(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
        return x

class Quake_Block(nn.Module):
    """
    Quake Block

    Convolutions used by Deepmind for Quake. Like original Nature CNN but uses
    more filters and skip connections.
    """
    def __init__(self, state_size):
        super().__init__(name='')

        prev_channels = state_size[-1]
        self.activation = nn.ReLU()
        self.conv2A = nn.Conv2d(in_channels=prev_channels, out_channels=32,
                                kernel_size=8, stride=4)
        self.conv2B = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=4, stride=2)
        self.conv2C = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1)
        self.conv2d = nn.Conv2D(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1)

    def forward(self, x):
        x = self.activation(self.conv2A(x))

        x = skip_1 = self.activation(self.conv2B(x))

        x = self.conv2C(x)
        x = skip_2 = x + skip_1
        x = self.activation(x)

        x = self.conv2D(x)
        x = x + skip_2
        x = self.activation(x)

        return x

if __name__ == "__main__":
    from src.general.envs.gym_env import GymEnv
    # env = GymEnv("BipedalWalker-v3")
    env = GymEnv("LunarLanderContinuous-v2")
    obs_space = env.obs_space
    action_space = env.action_space
    action_space_shape = get_action_space_shape(action_space)

    q = SACQNet(state_size=obs_space,
                action_space=action_space,
                fc=(128, 64),
                conv_size=None)
    actor = SACActor(state_size=env.obs_space,
                     action_space=env.action_space,
                     fc=(128, 64),
                     conv_size=None)

    test_q = q(torch.from_numpy(np.ones(obs_space.shape, dtype=np.float32)[None, :]),
               torch.from_numpy(np.ones(action_space_shape, dtype=np.float32)[None, :]))
    test_actor = actor(torch.from_numpy(np.ones(obs_space.shape, dtype=np.float32)[None, :]))
    print(f"Q OUT {test_q}")
    print(f"ACTOR OUT {test_actor}")
