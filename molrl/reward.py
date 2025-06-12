import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional
from abc import ABC

class FReward(ABC):
    """The base class for calculating rewards.

    Parameters
    ----------
    reward_fun_shape : str
        Shape of reward function. Available options: binary, linear, sigmoid and tanh.
    max_reward : float, optional
        Maximum value the reward can take
    min_reward : float, optional
        Minimum value the reward can take
    min_property_delta_cutoff : float, optional
        If the property change is above this cutoff, the reward takes the max reward value
    max_property_delta_cutoff : float, optional
        If the property change is below this cutoff, the reward takes the min reward value

    Examples
    --------
    Implementing the FReward base class with a custom reward function:

        ```python
        class QEDReward(FReward):
            def __init__(self, max_reward=0.0, min_reward=-1.0, min_property_delta_cutoff=0, max_property_delta_cutoff=0.1):
                super().__init__(
                max_reward=max_reward,
                min_reward=min_reward,
                min_property_delta_cutoff=min_property_delta_cutoff,
                max_property_delta_cutoff=max_property_delta_cutoff)

            def __call__(self, states: torch.Tensor, new_smiles: List[Union[str, None]], original_smiles: Optional[List[str]]) -> List[float]:
                R = []
                for smi in new_smiles:
                    # Calculate the QED score of the new smiles
                    qed = genchem.rl.utils.qed_score(smi)
                    R.append(qed)
                return R
        ```

    """

    def __init__(
        self,
        reward_fun_shape: str = None,
        max_reward: float = 0.0,
        min_reward: float = -1.0,
        min_property_delta_cutoff: float = 0,
        max_property_delta_cutoff: float = 0.1,
    ):
        if reward_fun_shape is None:
            raise ValueError(f'Reward function not specified')
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.min_property_delta_cutoff = min_property_delta_cutoff
        self.max_property_delta_cutoff = max_property_delta_cutoff
        if reward_fun_shape == 'binary':
            self.reward_fun = self.binary_reward
        elif reward_fun_shape == 'linear':
            self.reward_fun = self.linear_reward
        elif reward_fun_shape == 'sigmoid':
            self.reward_fun = self.sigmoid_reward
            self.max_reward = 0
            self.min_reward = -1
        elif reward_fun_shape == 'tanh':
            self.reward_fun = self.tanh_reward
            self.max_reward = 1
            self.min_reward = -1
        elif reward_fun_shape == 'delta':
            self.reward_fun = self.delta_reward
            self.max_reward = 2
            self.min_reward = -2
        elif reward_fun_shape == 'sampling':
            pass
        else:
            raise ValueError(f'Reward function {reward_fun_shape} not recognised.')

    def __call__(
        self,
        states: torch.Tensor,
        new_smiles: List[Union[str, None]],
        original_smiles: List[Union[str, None]],
    ) -> List[float]:
        """Given the state vectors, new smiles and original smiles, calculate the rewards

        Parameters
        ----------
        states : torch.Tensor

        new_smiles : List[Union[str, None]]

        original_smiles : List[Union[str, None]]

        Returns
        -------
        List[float]
            rewards
        """
        pass

    def binary_reward(self, df: pd.DataFrame):
        """
        Reward function.
        The reward is binary. It can take either the max or the min reward value,
        depending on the max_property_delta_cutoff.

        Parameters
        ----------
        df : pd.DataFrame
            pandas dataframe including a 'delta' column which is the property changes.

        Returns
        -------
        List[float]
            reward values
        """
        df['reward'] = 0
        df['reward'].loc[df['delta'] > self.max_property_delta_cutoff] = self.max_reward
        df['reward'].loc[df['delta'] <= self.max_property_delta_cutoff] = self.min_reward
        rewards = df.reward.values.astype(np.float32)
        rewards = np.clip(rewards, self.min_reward, self.max_reward).reshape(-1).tolist()
        return rewards

    def delta_reward(self, df: pd.DataFrame):
        """
        Reward function.
        The reward is equal to the difference.

        Parameters
        ----------
        df : pd.DataFrame
            pandas dataframe including a 'delta' column which is the property changes.

        Returns
        -------
        List[float]
            reward values
        """
        rewards = df.delta.values.astype(np.float32)
        return rewards

    def linear_reward(self, df: pd.DataFrame):
        """
        Reward function.
        The reward is linear in the region [min_property_delta_cutoff, max_property_delta_cutoff]
        and fixed beyond these areas.

        Parameters
        ----------
        df : pd.DataFrame
            pandas dataframe including a 'delta' column which is the property changes.

        Returns
        -------
        List[float]
            reward values
        """
        if self.min_property_delta_cutoff >= self.max_property_delta_cutoff:
            raise ValueError(f'Max reward cutoff must be greater than min reward cutoff.')
        slope = (abs(self.max_reward) + abs(self.min_reward)) / (
            self.max_property_delta_cutoff - self.min_property_delta_cutoff
        )
        df['reward'] = slope * df['delta'] + self.min_reward
        rewards = df.reward.values.astype(np.float32)
        rewards = np.clip(rewards, self.min_reward, self.max_reward).reshape(-1).tolist()
        return rewards

    def sigmoid_reward(self, df: pd.DataFrame):
        """
        Sigmoid reward function shifted by -1.
        """
        rewards = 1 / (1 + np.exp(((-1) * df['delta']).to_list())) - 1
        rewards = rewards.astype(np.float32)
        return rewards

    def tanh_reward(self, df: pd.DataFrame):
        """
        Tanh reward function.
        """
        rewards = np.tanh(df['delta'].to_list())
        rewards = rewards.astype(np.float32)
        rewards[rewards < 0] = self.min_reward
        return rewards

    def get_num_max_rewards(self, rewards: List[float]) -> int:
        '''Returns how many times the reward function takes max value'''
        num_max_rewards = np.sum(np.array(rewards) >= self.max_reward - 0.1)
        return num_max_rewards