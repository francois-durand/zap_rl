from typing import Union
import gym
import torch as th
from gym.spaces.discrete import Discrete

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule


class ZapQLPolicyDraft(BasePolicy):
    """
    Policy class with for tabular Zap Q-Learning

    Parameters
    ----------
    observation_space : gym.spaces.Space
        Observation space (must be discrete).
    action_space : gym.spaces.Space
        Action space (must be discrete).
    lr_schedule : Schedule
        Learning rate schedule (could be constant)
    a_hat_schedule : Schedule
        Learning rate schedule for a_hat (could be constant).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Union[float, Schedule],
        a_hat_schedule: Union[float, Schedule] = 0.01,
    ):
        super(ZapQLPolicyDraft, self).__init__(
            observation_space,
            action_space,
        )
        self.lr_schedule = lr_schedule
        self.a_hat_schedule = a_hat_schedule

        if not isinstance(observation_space, Discrete) or not isinstance(action_space, Discrete):
            raise ValueError("Both observation space and action space must be discrete.")
        self.n_observations = observation_space.n
        self.n_actions = action_space.n
        self.n_dims = observation_space.n * action_space.n

        self.q_tabular, self.q_tabular_target = None, None
        self.a_hat, self.a_hat_dagger = None, None
        self.zeta = None
        self._build()

    def _build(self) -> None:
        """
        Initialize the arrays.
        """
        self.q_tabular = th.zeros(self.n_observations, self.n_actions)
        self.q_tabular_target = th.zeros(self.n_observations, self.n_actions)
        self.a_hat = th.zeros(self.n_dims, self.n_dims)
        self.a_hat_dagger = th.zeros(self.n_dims, self.n_dims)
        self.zeta = th.zeros(self.n_observations, self.n_actions)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        action = self.q_tabular[obs, :].argmax(dim=1).reshape(-1)
        return action
