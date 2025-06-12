import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import logging
import time
import heapq
import wandb

from torch.distributions import MultivariateNormal
from typing import Optional, List, Dict

from tqdm import tqdm
from .latent import LatentModel
from .reward import FReward

logging.basicConfig(level=logging.INFO)

### Utility functions ###
def reinitialize_weights(model):
    """Reinitialize the weights of a given model."""
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def set_seed(seed: int):
    """Set all seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If using CUDA, enable deterministic algorithms and set the CUDA seed.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Ensures reproducibility by disabling auto-tuning


### Reward helper ###
class BaseRewardSimple:
    """
    Reward function for evaluating the reward of a given action
    """

    def __init__(self, calculator) -> None:
        self.calc = calculator

    def __call__(self, action, new_smiles, original_smiles):
        R = []
        for i in range(len(new_smiles)):
            val = self.calc(new_smiles[i])
            R.append(val)
        return R


### ActorCritic architecture ###
class ActorCritic(nn.Module):
    """
    Actor Critic Network for Proximity Policy Optimization (PPO)

    Parameters
    ----------
    input_dim : int
        Input dimension of the network
    action_dim : int
        Action dimension of the network
    hidden_dim : int
        Hidden dimension of the network

    """

    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.shared_layer(x)
        policy_mean = self.actor(x)
        value = self.critic(x)
        return policy_mean, 0, value



class Hyperparameters:
    """
    Container for managing hyperparameters. Stores hyperparameters in a dictionary.
    """

    def __init__(self):
        # Start with an empty dictionary
        self._hyperparameters = {}

    def __getattr__(self, name):
        if name in self._hyperparameters:
            return self._hyperparameters[name]
        raise AttributeError(f"'Hyperparameters' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name != "_hyperparameters":
            self._hyperparameters[name] = value
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        return repr(self._hyperparameters)


### PPOAgent Class ###
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for Reinforcement Learning in latent space of an autoencoder.

    Parameters
    ----------
    latent_model : LatentModel
        Latent model for encoding and decoding SMILES strings
    policy_net_latent : torch.nn.Module
        Policy network for learning the optimal policy in latent space
    reward_fn : genchem.rl.FReward
        Reward function for evaluating the reward of a given action
    device : torch.device
        Device to run the model on


    Examples
    --------
    import molrl
    import logging
    from rdkit.Chem import QED, MolFromSmiles

    def calculate_qed(smiles):
        try:
            return QED.qed(MolFromSmiles(smiles))
        except:
            return 0

    reward_fn = molrl.BaseRewardSimple(calculator=calculate_qed)
    vae = molrl.VAEModel()
    actor_critic = molrl.ActorCritic(
            input_dim=vae.get_latent_size(),
            action_dim=vae.get_latent_size(),
            hidden_dim=vae.get_latent_size(),
    )


    agent = molrl.PPOAgent(
            latent_model=vae,
            policy_net_latent=actor_critic,
            reward_fn=reward_fn,
            device="cuda:0",
    )

    agent.train(
        epochs=10,
        random_seed=42,
        seed=0,
        latent_size=64,
        batch_size=10,
        deterministic=True,
    )
    """

    def __init__(
        self,
        latent_model: LatentModel = None,
        policy_net_latent: ActorCritic = None,
        reward_fn: FReward = None,
        device: torch.device = None,
        verbosity: Optional[int] = logging.INFO,
        **kwargs,
    ):
        self.latent_model = latent_model
        if "vae_model" in kwargs:
            self.latent_model = kwargs["vae_model"]
        if "mode" in kwargs:
            self.mode = kwargs["mode"]
        self.policy_net_latent = policy_net_latent
        self.device = device
        self.reward_fn = reward_fn
        self.verbosity = verbosity

        # Set up the logging configuration based on the verbosity level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.verbosity)

        self.logger.info(f"Initialized PPOAgent on device: {self.device}")

        self.hp = Hyperparameters()

        # Handle deprecated parameters during class instantiation
        self._handle_deprecated_params(kwargs, during_init=True)
        self._convert_latent_model()

        self.policy_net_latent.to(self.device)
        self.latent_model.to(self.device)

        # Inspection hooks for debugging
        self.hooks = {}

    def _handle_deprecated_params(self, kwargs, during_init=False):
        # Check for deprecated parameters defined at class instantiation and handle renaming
        renamed_params = {
            "lr": "learning_rate",
            "k": "n_top_k",
            "random_seed": "n_random_seeds",
            "project_name": "wandb_project_name",
        }

        if "mode" in kwargs:
            self.logger.warning("The `mode` argument is deprecated, use a LatentModel instead.")
            self.mode = kwargs["mode"]

        for deprecated_param in [
            "lr",
            "mode",
            "clip_epsilon",
            "l2_ratio",
            "update_steps",
            "save_top_k",
            "entropy_coef",
            "k",
            "disable_ppo",
            "random_seed",
            "project_name",
        ]:
            if deprecated_param in kwargs:
                if during_init:
                    self.logger.warning(
                        f"Passing `{deprecated_param}` during class instantiation is deprecated, pass to `train` method instead."
                    )
                if deprecated_param in renamed_params:
                    self.logger.warning(
                        f"Using `{deprecated_param}` is deprecated, use `{renamed_params[deprecated_param]}` instead."
                    )
                    setattr(self.hp, renamed_params[deprecated_param], kwargs[deprecated_param])
                else:
                    setattr(self.hp, deprecated_param, kwargs[deprecated_param])

        if "optimization" in kwargs:
            self.logger.warning(
                "The `optimization` argument is deprecated, use `state_update_method` instead."
            )
            if kwargs["optimization"]:
                setattr(self.hp, "state_update_method", "scaled")
            else:
                setattr(self.hp, "state_update_method", "additive")

        if "vae_model" in kwargs:
            self.logger.warning(
                "The `vae_model` argument is deprecated, use `latent_model` instead."
            )
            self.latent_model = kwargs["vae_model"]

    def _convert_latent_model(self):
        if not isinstance(self.latent_model, LatentModel):
            self.logger.warning(
                f"Directly passing a model  of type: {type(self.latent_model)} is deprecated. Use a LatentModel instead."
            )
            self.latent_model = LatentModel.to_latent_model(self.latent_model)

    def _evaluate_action(self, original_smiles, action_vector, return_smiles=False):
        """Given an action vector, evaluate the action vector and return the reward"""
        new_smiles = self.latent_model.decode(action_vector)
        rewards = self.reward_fn(action_vector, new_smiles, original_smiles)
        rewards = torch.tensor(rewards).to(device=self.device).float()
        if return_smiles:
            return rewards, new_smiles
        else:
            return rewards

    def train(
        self,
        # Strategy and modes
        training_strategy: str = "ppo",
        disable_ppo: bool = False,
        state_update_method: str = "additive",
        # Input data and initialization
        smiles: Optional[List[str]] = None,
        n_random_seeds: Optional[int] = None,
        # QUESTION: Is this used?
        original_states: Optional[torch.Tensor] = None,
        # Learning and optimization parameters
        learning_rate: float = 1e-4,
        l2_ratio: float = 0.0,
        clip_epsilon: float = 0.2,
        update_steps: int = 32,
        batch_size: int = 1,
        steps: int = 1,
        n_top_k: int = 10,
        save_top_k: int = None,
        early_stopping_k: int = None,
        early_stopping_tolerance: float = 1e-3,
        # Entropy and exploration
        entropy_coef: float = 1e-5,
        fix_std: bool = True,
        std: float = 0.1,
        std_decay_rate: float = 0.9995,
        min_std: float = 0.1,
        std_reset_epochs: int = 4,
        reset_std: bool = False,
        # Training iterations
        epochs: int = 10,
        reset_to_best_state: Optional[int] = None,
        reset_each_epochs: int = 10000,
        # Reward-related parameters
        normalize_rewards: bool = True,
        # Logging and tracking
        log_progress: bool = True,
        enable_wandb: bool = False,
        wandb_project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        # Randomness and seeding
        seed: int = 0,
        deterministic: bool = False,
        save_states_and_actions: bool = False,
        # Additional kwargs
        **kwargs: Dict,
    ) -> "PPOAgent":
        """
        Train the PPO agent for reinforcement learning in latent space.

        Parameters
        ----------
        training_strategy : str, optional
            The training strategy to use. Default is 'ppo'. Other strategies
            might include 'reinforce'.

        disable_ppo : bool, optional
            If True, disables the PPO update steps and uses a vanilla policy gradient.
            Default is False.

        state_update_method : str, optional
            Defines how the action is applied to the state during training.
            The available options are:

            - "scaled": The action is used to scale the current state, i.e., the action is multiplied by the current state and then added.
            - "additive": The action is directly added to the current state (standard approach in reinforcement learning).

            Default is "additive".

        smiles : list, optional
            A list of SMILES strings representing molecules. If not provided, random seeds
            will be used to generate initial states.

        original_states : torch.Tensor, optional
            A tensor representing the initial states to start from in latent space. This is
            typically encoded from SMILES strings.

        learning_rate : float, optional
            The learning rate for the optimizer. Default is 1e-4.

        l2_ratio : float, optional
            The L2 regularization coefficient. Default is 0.0.

        clip_epsilon : float, optional
            The clipping parameter for PPO to prevent large updates. Default is 0.2.

        update_steps : int, optional
            The number of PPO update steps per exploration batch. Default is 32.

        batch_size : int, optional
            The size of the batch during training. Default is 1.

        steps : int, optional
            The number of steps per exploration batch. Default is 1.

        n_top_k : int, optional
            The number of top results to track and save per exploration batch. Default is 10.

        save_top_k : int, optional
            The number of top k results to save per exploration batch. If None, results will not be saved. Default is None.

        early_stopping_k : int, optional
            The number of consecutive episodes with the same reward value to trigger early stopping.
            If None, early stopping is disabled. Default is None.

        entropy_coef : float, optional
            The coefficient for entropy in the loss function, encouraging exploration.
            Default is 1e-5.

        fix_std : bool, optional
            If True, fixes the standard deviation of the action distribution. Default is True.

        std : float, optional
            The standard deviation for the action distribution. Default is 0.1.

        std_decay_rate : float, optional
            The decay rate for the standard deviation over time. Default is 0.9995.

        min_std : float, optional
            The minimum allowed value for the standard deviation. Default is 0.1.

        std_reset_epochs : int, optional
            The number of epochs after which the standard deviation is reset. Default is 4.

        reset_std : bool, optional
            If True, resets the standard deviation to the initial value after a fixed number
            of epochs. Default is False.

        epochs : int, optional
            The number of training epochs. Default is 10.

        reset_to_best_state : int, optional
            If provided, resets the training to the best performing state after this many epochs.
            Default is None.

        reset_each_epochs : int, optional
            The number of epochs after which to reset to the best state or reinitialize.
            Default is 10000.

        normalize_rewards : bool, optional
            If True, normalizes the rewards to improve training stability. Default is True.

        log_progress : bool, optional
            If True, logs training metrics during training using tqdm. Default is True.

        enable_wandb : bool, optional
            If True, enables Weights and Biases logging for experiment tracking. Default is False.

        wandb_project_name : str, optional
            The project name for Weights and Biases logging. Default is None.

        run_name : str, optional
            The run name for Weights and Biases logging. Default is None.

        n_random_seeds : int, optional
            If provided, seeds the random number generator for reproducibility.
            Default is None.

        seed : int, optional
            The seed for PyTorch's random number generator. Default is 0.

        deterministic : bool, optional
            If True, ensures deterministic behavior during training by fixing the action
            selection and random seeds. Default is False.

        **kwargs : dict
            Additional parameters for flexibility.

        Returns
        -------
        PPOAgent
            The trained agent with updated policy and value networks.

        Notes
        -----
        - If both `smiles` and `original_states` are not provided, the agent will initialize
          random latent states using the `n_random_seeds` or `seed` values.
        - If `disable_ppo` is True, the agent will perform vanilla policy gradient updates
          instead of the clipped PPO updates.
        - Set `enable_wandb=True` to log training progress to Weights and Biases.
        - Early stopping will occur if `early_stopping_k` is set, and the reward has not changed
          for the last `early_stopping_k` episodes.
        """
        # Store hyperparameters in a dictionary
        self.hp.training_strategy = training_strategy
        self.hp.disable_ppo = disable_ppo
        self.hp.state_update_method = state_update_method
        self.hp.n_random_seeds = n_random_seeds

        self.hp.learning_rate = learning_rate
        self.hp.l2_ratio = l2_ratio
        self.hp.clip_epsilon = clip_epsilon
        self.hp.update_steps = update_steps
        self.hp.batch_size = batch_size
        self.hp.steps = steps
        self.hp.n_top_k = n_top_k
        self.hp.save_top_k = save_top_k
        self.hp.early_stopping_k = early_stopping_k
        self.hp.early_stopping_tolerance = early_stopping_tolerance
        self.hp.entropy_coef = entropy_coef
        self.hp.fix_std = fix_std
        self.hp.std = std
        self.std = self.hp.std
        self.hp.std_decay_rate = std_decay_rate
        self.hp.min_std = min_std
        self.hp.std_reset_epochs = std_reset_epochs
        self.hp.reset_std = reset_std

        self.hp.epochs = epochs
        self.hp.reset_to_best_state = reset_to_best_state
        self.hp.reset_each_epochs = reset_each_epochs

        self.hp.normalize_rewards = normalize_rewards

        self.log_progress = log_progress
        self.hp.enable_wandb = enable_wandb
        self.hp.wandb_project_name = wandb_project_name
        self.hp.run_name = run_name

        self.hp.seed = seed
        self.hp.deterministic = deterministic

        self.smiles = smiles
        self.original_states = original_states
        self.run_name = run_name
        self.save_states_and_actions = save_states_and_actions
        self.states = [] 
        self.actions = []

        self._handle_deprecated_params(kwargs)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net_latent.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, verbose=True)
        if self.hp.deterministic:
            set_seed(seed)

        # QUESTION: Is this the right way to disable PPO? Do we want to raise a warning if the user sets update_steps and disable_ppo?
        if self.hp.disable_ppo:
            self.logger.debug('PPO is disabled')
            self.update_steps = 1

        self.train_metrics = []
        self.seen_buffer = {}

        # QUESTION: What is this used for?
        self.total_reward_calls = batch_size * steps * epochs

        # Initialize the agent for optimization
        self._initialize_optimization()

        # Initialize Weights and Biases logging
        if enable_wandb:
            self._init_wandb(wandb_project_name, run_name)

        self._train()

    def _initialize_optimization(self):
        """Initialize the optimization using the provided SMILES / original states or using random seeds."""
        if self.smiles is not None:
            self.logger.debug("Initializing optimization using provided SMILES")
            self.smiles = np.array(self.smiles)
            self.original_states = self.latent_model.encode(self.smiles)

        elif self.hp.n_random_seeds is not None:
            self.logger.debug("Initializing optimization using random seeds")
            # Handle different kinds of random seeds
            if self.hp.deterministic:
                set_seed(self.hp.seed)
            elif self.hp.seed is not None:
                torch.manual_seed(self.hp.seed)

            self.original_states = torch.randn(
                self.hp.n_random_seeds, self.latent_model.get_latent_size()
            ).to(self.device)

            self.smiles = self.latent_model.decode(self.original_states)

        elif self.original_states is not None:
            self.logger.debug("Initializing optimization using provided original_states")
            self.original_states = self.original_states.to(self.device)
            self.smiles = self.latent_model.decode(self.original_states)

        else:
            if (
                self.smiles is None
                and self.hp.n_random_seeds is None
                and self.original_states is None
            ):
                raise ValueError(
                    'Either smiles, original_states, or n_random_seeds must be provided'
                )

        self.n_initial_states = self.original_states.shape[0]

        # TODO: is this the best way to store the top k results?
        self.top_per_sample = [
            [] for _ in range(self.n_initial_states)
        ]  # init for top_k results during exploration

    def _init_wandb(self, wandb_project_name, run_name):
        if wandb_project_name is None:
            wandb_project_name = 'PPO-rl'
        if run_name is None:
            wandb.init(entity='cellarity-ai', project=wandb_project_name, config=self.hp)
        else:
            wandb.init(
                entity='cellarity-ai', project=wandb_project_name, config=self.hp, name=run_name
            )

        wandb.watch(self.policy_net_latent)

    def _sample_action(self, policy_mean, std):
        """Sample action from policy distribution."""
        if self.hp.deterministic:
            set_seed(self.hp.seed)

        action_var = torch.full((self.latent_model.get_latent_size(),), std**2).to(self.device)
        normal_dist = torch.distributions.MultivariateNormal(
            policy_mean, torch.diag(action_var).unsqueeze(dim=0)
        )
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action)
        return action, log_prob

    def _get_next_action(self, original_states, action):
        """Determine the next action based on the current state and mode."""
        if self.hp.state_update_method == "scaled":
            return original_states + (original_states * action)
        elif self.hp.state_update_method == "additive":
            return original_states + action
        elif self.hp.state_update_method == "replace":
            return action
        else:
            raise ValueError(
                f"Invalid state update method: {self.hp.state_update_method}. Must be in ['scaled', 'additive']"
            )

    def _train(
        self,
    ):
        
        """Run the Proximal Policy Optimization (PPO) training loop."""
        self.early_stop = False

        start_time = time.time()
        if self.log_progress:
            self._pbar = tqdm(total=self.hp.epochs, desc="Training", position=0, leave=True)

        for epoch in range(self.hp.epochs):
            if self._check_early_stopping():
                break

            rewards_sum = 0.0
            old_log_probs, old_values, states_samples, actions_samples, rewards_samples = (
                [],
                [],
                [],
                [],
                [],
            )

            if self.hp.deterministic:
                set_seed(self.hp.seed)

            sample_indices = torch.randint(0, self.n_initial_states, (self.hp.batch_size,)).to(
                self.device
            )

            self.trigger_hook("epoch_init", sample_indices, self.original_states)

            # Collect experiences by interacting with the environment
            for _ in range(self.hp.steps):
                if self.hp.deterministic:
                    set_seed(self.hp.seed)

                # Sample starting states and smiles
                original_states_sample = torch.index_select(self.original_states, 0, sample_indices)
                smiles_sample = self.smiles[sample_indices.cpu().numpy()]

                # Forward pass through the policy network
                policy_mean, _, value = self.policy_net_latent(original_states_sample)

                # Action sampling
                action, log_prob = self._sample_action(policy_mean, self.std)

                if self.save_states_and_actions:
                    self.states.append(original_states_sample)
                    self.actions.append(action)

                # Next state based on action
                next_action = self._get_next_action(original_states_sample, action)

                # Evaluate actions and get rewards
                rewards, new_smiles = self._evaluate_action(
                    smiles_sample, next_action, return_smiles=True
                )

                # Update buffer
                self._update_buffer(
                    new_smiles,
                    smiles_sample,
                    sample_indices,
                    rewards,
                    epoch,
                )

                # Collect relevant samples and metrics
                old_log_probs.append(log_prob.detach())
                old_values.append(value.detach())
                states_samples.append(original_states_sample.detach())
                actions_samples.append(action.detach())
                rewards_samples.append(rewards.clone().detach().to(self.device).float())

                rewards_sum += rewards.mean()

            # Perform PPO update
            if self.hp.training_strategy == "ppo":
                self._update_ppo(
                    states_samples,
                    actions_samples,
                    old_log_probs,
                    old_values,
                    rewards_samples,
                    rewards_sum,
                    epoch,
                )
            elif self.hp.training_strategy == "reinforce":
                self._reinforce(log_prob, rewards, rewards_sum, epoch)
            else:
                raise ValueError(
                    f"Invalid training strategy: {self.hp.training_strategy}. Must be in ['ppo', 'reinforce']"
                )

            # Adjust learning parameters, log results, and reset if necessary
            self._log_and_reset(epoch)

        self.run_duration = time.time() - start_time
        if self.hp.enable_wandb:
            wandb.finish()
        if self.log_progress:
            self._pbar.close()
        return self

    def _update_ppo(
        self,
        states_samples,
        actions_samples,
        old_log_probs,
        old_values,
        rewards_samples,
        rewards_sum,
        epoch,
    ):
        """Perform PPO update and log all required metrics."""
        old_log_probs = torch.cat(old_log_probs)
        old_values = torch.cat(old_values)
        states_samples = torch.cat(states_samples)
        actions_samples = torch.cat(actions_samples)
        returns = torch.cat(rewards_samples).flatten()

        if self.hp.normalize_rewards:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        advantages = returns - old_values.flatten()

        # Initialize tracking variables for additional metrics
        avg_loss = 0.0
        l2_loss_sum = 0.0
        oracle_calls = self.hp.batch_size * self.hp.steps * (epoch + 1)

        for _ in range(self.hp.update_steps):
            mu, _, values = self.policy_net_latent(states_samples)
            normal_dist = MultivariateNormal(
                mu,
                torch.diag(torch.full((self.latent_model.get_latent_size(),), self.std**2)).to(
                    self.device
                ),
            )
            new_log_probs = normal_dist.log_prob(actions_samples)
            entropy = normal_dist.entropy().mean()

            # Calculate the ratio and PPO surrogate losses
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.hp.clip_epsilon, 1.0 + self.hp.clip_epsilon)
                * advantages
            ) + 1e-5

            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean() 
            critic_loss = torch.nn.functional.mse_loss(values.flatten(), returns)

            # Total loss includes L2 regularization (if used), and PPO update steps
            batch_loss = 0.5 * critic_loss + actor_loss - self.hp.entropy_coef * entropy

            # Calculate L2 loss for tracking
            l2_loss = torch.sum(actions_samples * actions_samples) / actions_samples.size(0)
            l2_loss_sum += l2_loss.item()

            avg_loss += batch_loss.item()

            # Perform backpropagation and update
            if not self.hp.disable_ppo:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

        # Gather all relevant metrics for logging
        metrics = {
            'episode_reward': rewards_sum.item(),
            'loss': avg_loss / self.hp.update_steps,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'l2_loss': l2_loss_sum / self.hp.update_steps,
            'epoch': epoch,
            'action_std': self.std,
            'oracle_calls': oracle_calls,
            'mean_top_k_reward': self.get_top_k_rewards(k=self.hp.n_top_k),
            'mean_top_k_unique': self.get_top_k_rewards_unique(k=self.hp.n_top_k),
            'max_reward': self.get_max_reward().item(),
        }

        self.train_metrics.append(metrics)
        
        if self.log_progress:
            self._pbar.set_postfix(metrics)
            self._pbar.update(1)
        else:
            self.logger.debug(metrics)

        # Log metrics with wandb if enabled
        if self.hp.enable_wandb:
            wandb.log(metrics)

    def _reinforce(self, log_prob, rewards, rewards_sum, epoch):
        """Perform REINFORCE update and log all required metrics."""
        # Update policy (REINFORCE)
        policy_loss = (-log_prob * rewards).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Standard deviation decay and logging
        original_std = self.std
        self.std = max(self.hp.min_std, self.std * self.hp.std_decay_rate)
        self.logger.debug(f"Decaying standard deviation from: {original_std} to {self.std}.")
        self._log_and_reset(epoch)

        metrics = {
            'episode_reward': rewards_sum.item(),
            'loss': policy_loss.item(),
            'epoch': epoch,
            'action_std': self.std,
            'mean_top_k_reward': self.get_top_k_rewards(k=self.hp.n_top_k),
        }

        self.train_metrics.append(metrics)

        if self.log_progress:
            self._pbar.set_postfix(metrics)
            self._pbar.update(1)
        else:
            self.logger.debug(metrics)

        if self.hp.enable_wandb:
            wandb.log(metrics)

    def _log_and_reset(self, epoch):
        """Handle logging and resetting hyperparameters during training."""
        original_std = self.std
        if (epoch + 1) % self.hp.std_reset_epochs == 0 and self.hp.reset_std:
            self.logger.debug('Resetting standard deviation to initial value.')
            self.std = self.hp.min_std
        elif (
            self.hp.reset_to_best_state is not None
            and (epoch + 1) % self.hp.reset_to_best_state == 0
        ):
            self.logger.debug(f"Resetting to best states at epoch {epoch + 1}.")
            # Reset to best states based on top-k results
            self._reset_original_states(n_states=self.hp.n_top_k)
        else:
            # Decay the standard deviation gradually
            self.std = max(self.hp.min_std, self.std * self.hp.std_decay_rate)
            self.logger.debug(f"Decaying standard deviation from: {original_std} to {self.std}.")

    def _check_early_stopping(self):
        """
        Checks if early stopping should be triggered based on the last k episode rewards.

        If the rewards in the last k episodes are within a specified tolerance,
        early stopping is triggered. If tolerance is set to 0, the rewards must
        be exactly the same for early stopping to occur (mimics previous behavior).

        Returns
        -------
        bool
            True if early stopping conditions are met, False otherwise.
        """
        if self.hp.early_stopping_k is None or len(self.train_metrics) < self.hp.early_stopping_k:
            return False

        last_k_rewards = [
            m['episode_reward'] for m in self.train_metrics[-self.hp.early_stopping_k :]
        ]

        # If the rewards are all within the tolerance range, trigger early stopping
        if max(last_k_rewards) - min(last_k_rewards) <= self.hp.early_stopping_tolerance:
            self.logger.debug(
                f"Early stopping condition met with tolerance: {self.hp.early_stopping_tolerance}. "
                "Ending training loop early."
            )
            return True

        return False

    def train_reinforce(
        self,
        smiles: Optional[List[str]] = None,
        batch_size: int = 1,
        steps: int = 1,
        std: float = 0.1,
        epochs: int = 10,
        normalize_rewards: bool = False,
        original_states: Optional[torch.Tensor] = None,
        std_decay_rate: float = 0.9995,
        min_std: float = 0.1,
        random_seed: Optional[int] = None,
        seed: int = 0,
        deterministic: bool = False,
        **kwargs: Dict,
    ):
        """
        Train the agent using the REINFORCE algorithm.

        Parameters
        ----------
        smiles : list, optional
            List of SMILES strings to initialize states from.
        batch_size : int, optional
            The number of samples per batch. Default is 1.
        steps : int, optional
            Number of steps in each epoch. Default is 1.
        std : float, optional
            Initial standard deviation for action sampling. Default is 0.1.
        epochs : int, optional
            Number of training epochs. Default is 10.
        normalize_rewards : bool, optional
            Normalize rewards for stability. Default is False.
        original_states : torch.Tensor, optional
            Initial latent states. If not provided, will be generated.
        latent_size : int, optional
            The latent space size. Default is 512.
        std_decay_rate : float, optional
            Decay rate for standard deviation. Default is 0.9995.
        min_std : float, optional
            Minimum value for the standard deviation. Default is 0.1.
        random_seed : int, optional
            If provided, use this seed to generate random latent states.
        seed : int, optional
            Random seed for deterministic behavior. Default is 0.
        deterministic : bool, optional
            If True, ensures deterministic behavior during training.
        **kwargs : dict
            Additional hyperparameters.

        Returns
        -------
        self
            Trained agent.
        """
        self.logger.debug(
            "Using .train_reinforce() method is deprecated, use .train(... training_strategy='reinforce') instead."
        )

        self.train(
            training_strategy="reinforce",
            smiles=smiles,
            original_states=original_states,
            batch_size=batch_size,
            steps=steps,
            std=std,
            epochs=epochs,
            normalize_rewards=normalize_rewards,
            std_decay_rate=std_decay_rate,
            min_std=min_std,
            random_seed=random_seed,
            seed=seed,
            deterministic=deterministic,
            **kwargs,
        )

        return self

    def _reset_original_states(self, n_states: int = 10, method: str = "top_k"):
        """Reset the original states to the best states or random states."""
        if method == "top_k":
            self.logger.debug("Resetting original states to best states.")
            results = self.buffer()
            results = results.drop_duplicates(subset='new_smiles')
            results = results.sort_values('reward', ascending=False)
            self.smiles = results['new_smiles'][0 : self.n_top_k].values
            self.original_states = self.latent_model.encode(self.smiles)
        elif method == "random":
            self.logger.debug("Resetting original states to random states.")
            self.original_states = torch.randn(n_states, self.latent_model.get_latent_size()).to(
                self.device
            )
            self.smiles = self.latent_model.decode(self.original_states)
        return

    def _update_buffer(
        self,
        new_smiles,
        smiles_sample,
        sample_indices,
        rewards,
        epoch=0,
        steps=None,
        batch_size=None,
    ):
        if steps is None:
            steps = self.hp.steps

        if batch_size is None:
            batch_size = self.hp.batch_size

        if self.hp.state_update_method == "scaled":
            heap_index = sample_indices
        else:
            heap_index = None

        # Store top k results per sample
        for index_select, (new_smi, original_smi, sample_idx, reward) in enumerate(
            zip(new_smiles, smiles_sample, sample_indices, rewards)
        ):
            if heap_index is None:
                sample_idx = 0  # if heap index is not provided, store in first heap, else store in each instance of heap

            self._store_value_in_buffer(
                self.top_per_sample,
                (new_smi, original_smi, reward),
                sample_idx,
                max_size=self.hp.save_top_k,
                step=index_select + (epoch * steps * batch_size),
            )

    def save(self, model_path):
        # Save model
        torch.save(self.policy_net_latent.state_dict(), model_path)

    def load(self, model_path):
        # Load model
        self.policy_net_latent.load_state_dict(torch.load(model_path))
        return self

    def buffer(self):
        dfs = [pd.DataFrame(i) for i in self.top_per_sample]
        r = pd.concat(dfs)
        r.columns = ['reward', 'new_smiles', 'original_smiles', 'oracle_calls']
        return r

    def get_top_k_rewards(self, k=10):
        try:
            r = self.buffer()
            return r['reward'].nlargest(k).mean()
        except:
            return 0

    def get_top_k_rewards_unique(self, k=10):
        try:
            r = self.buffer()
            r = r.drop_duplicates(subset='new_smiles')
            return r['reward'].nlargest(k).mean()
        except:
            return 0

    def get_max_reward(self):
        try:
            r = self.buffer()
            return r['reward'].max()
        except:
            return 0

    def _store_value_in_buffer(self, heap, value, val_idx, max_size=None, step=0):
        new_smi, original_smi, reward = value
        if new_smi is None:
            new_smi = ''  # heap doesnt accept None
        if reward is None:
            reward = 0.0

        sub_heap = heap[val_idx]

        if original_smi is None:
            original_smi = ''
        heapq.heappush(sub_heap, (reward.item(), new_smi, original_smi, step))
        # check if you want to save the entire training buffer or only top k per starting trajectory
        if max_size is not None:
            if len(sub_heap) > max_size:  # keep only top k results, else save all results in buffer
                heapq.heappop(sub_heap)

    def results(self):
        return self.buffer()

    def save_buffer(self, path):
        self.buffer().to_csv(path, index=False)

    def get_train_results(self):
        """Convert an agent object to an anndata object"""
        import anndata as ad

        adata = ad.AnnData(self.buffer()[["reward", "oracle_calls"]])
        if hasattr(self, "states"):
            state_names = self.states[0].keys()
            for state_name in state_names:
                adata.obsm[state_name] = np.concatenate(
                    [epoch[state_name] for epoch in self.states]
                )

        adata.uns["train_metrics"] = pd.DataFrame(self.train_metrics)
        adata.uns["run_duration"] = self.run_duration
        buffer = self.buffer()
        buffer.index = buffer.index.astype(str)
        adata.obs = buffer

        return adata

    def save_policy(self, path):
        torch.save(self.policy_net_latent.state_dict(), path)

    def load_policy(self, path):
        self.policy_net_latent.load_state_dict(torch.load(path))
        return self

    def add_hook(self, name: str, hook_fn):
        """Register a hook function for a specific point in the training loop."""
        self.hooks[name] = hook_fn

    def trigger_hook(self, name: str, *args):
        """Trigger the hook if it has been registered."""
        if name in self.hooks:
            self.hooks[name](*args)