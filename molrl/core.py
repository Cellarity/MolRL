import molrl
from molrl import ActorCritic, PPOAgent, BaseRewardSimple

def create_default_agent(reward_fn:callable, device='cuda:0') -> PPOAgent:
    """
    Create a default PPO agent with a VAE model and a simple reward function.

    Parameters:
    -----------
    reward_fn : callable
        A function that takes a batch of SMILES strings and returns rewards.
    device : str
        The device to run the agent on, default is 'cuda:0'.
    
    Returns:
    --------
    PPOAgent
        An instance of the PPOAgent with the specified VAE model and reward function.

    """
    vae = molrl.VAEModel(device=device)
    reward_fn = BaseRewardSimple(reward_fn)
    ls = 64
    network = ActorCritic(ls,ls,ls)
    agent = PPOAgent(vae, network, reward_fn=reward_fn, device=device)
    return agent