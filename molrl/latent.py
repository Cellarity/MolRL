import torch
import logging
import numpy as np
from .rnn import SMILESVAERNNCAT

class LatentModel:
    """
    Abstract Latent Model class.

    Provides a unified interface for handling latent encoding and decoding across different models.
    """
    def to_latent_model(model):
        """
        Factory method to return the appropriate latent model instance.
        """
        if isinstance(model, SMILESVAERNNCAT):
            return VAEModel(model)
        else:
            from bionemo.model.molecule.molmim.infer import MolMIMInference

            if isinstance(model, MolMIMInference):
                return MolMIMModel(model)
            else:
                raise NotImplementedError(
                    f"Could not automatially create LatentModel from model type: {type(model)}"
                )
    def to_list(self, smiles_list):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        return smiles_list

    def to(self, device: torch.device, **kwargs):
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            Device to move the model to.
        """
        self.model.to(device, **kwargs)

    def encode(self, smiles_list, use_batching=True, batch_size=1000):
        """
        Encode SMILES strings to latent space with optional batching.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings to encode.
        use_batching : bool, optional
            Whether to use batching. If False, the entire input is processed at once. Default is True.
        batch_size : int, optional
            Number of SMILES strings to encode at once if batching is enabled. Default is 1000.

        Returns
        -------
        torch.Tensor
            Latent space representations of the input SMILES strings.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def decode(self, latent_vectors, use_batching=True, batch_size=1000):
        """
        Decode latent space vectors back to SMILES strings with optional batching.

        Parameters
        ----------
        latent_vectors : torch.Tensor
            Latent vectors to decode into SMILES strings.
        use_batching : bool, optional
            Whether to use batching. If False, the entire input is processed at once. Default is True.
        batch_size : int, optional
            Number of latent vectors to decode at once if batching is enabled. Default is 1000.

        Returns
        -------
        list of str
            Decoded SMILES strings.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_latent_size(self):
        """
        Returns the dimensionality of the latent space for this model.

        Returns
        -------
        int
            Dimensionality of the latent space.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class VAEModel(LatentModel):
    """
    VAE Model class, implementing the LatentModel interface.
    """

    def __init__(self, model:SMILESVAERNNCAT,device='cuda:0'):
        self.model = model

    def encode(self, smiles_list, use_batching=True, batch_size=1000):
        """
        Encode SMILES strings using the VAE model with optional batching.
        """
        smiles_list = self.to_list(smiles_list)
        if use_batching and len(smiles_list) > batch_size:
            all_latent_vectors = []
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i : i + batch_size]
                latent_vectors = self.model.smiles_to_latent(batch_smiles, return_cpu=False)
                all_latent_vectors.append(torch.tensor(latent_vectors).float())
            return torch.cat(all_latent_vectors)
        else:
            latent_vectors = self.model.smiles_to_latent(smiles_list, return_cpu=False)
            return torch.tensor(latent_vectors).float().to(self.model.device)

    def decode(self, latent_vectors, use_batching=True, batch_size=1000):
        """
        Decode latent vectors back to SMILES using the VAE model with optional batching.
        """
        if use_batching and len(latent_vectors) > batch_size:
            all_smiles = []
            for i in range(0, len(latent_vectors), batch_size):
                batch_latents = latent_vectors[i : i + batch_size]
                batch_smiles = self.model.latent_to_smiles_batched(
                    batch_latents, argmax=True, total=1
                )
                all_smiles.extend(batch_smiles)
            return torch.tensor(all_smiles).to(self.model.device)
        else:
            return np.array(
                self.model.latent_to_smiles_batched(latent_vectors, argmax=True, total=1)
            )

    def get_latent_size(self):
        """
        Return the latent size of the VAE model.
        """
        return self.model.latent_size

class MolMIMModel(LatentModel):
    """
    MolMIM Model class, implementing the LatentModel interface.
    """

    def __init__(
        self,
        model=None,
        BIONEMO_HOME="/workspace/bionemo",
        MOLMIM_CHECKPOINTS_PATH="/home/jovyan/workspace/bionemo/molmim_70m_24_3.nemo",
    ):
        if model is not None:
            self.model = model
        else:
            self.model = self.load_molmim(BIONEMO_HOME, MOLMIM_CHECKPOINTS_PATH)

    def load_molmim(
        self,
        BIONEMO_HOME="/workspace/bionemo",
        MOLMIM_CHECKPOINTS_PATH="/home/jovyan/workspace/bionemo/molmim_70m_24_3.nemo",
    ):
        import os
        from bionemo.utils.hydra import load_model_config
        from bionemo.model.molecule.molmim.infer import MolMIMInference

        config_path = os.path.join(BIONEMO_HOME, "examples/tests/conf/")

        logging.info(f"Loading MolMIM config from: {config_path}")
        cfg = load_model_config(
            config_name="molmim_infer.yaml",
            config_path=config_path,
        )

        logging.info(f"Loading MolMIM checkpoints from: {MOLMIM_CHECKPOINTS_PATH}")
        cfg.model.downstream_task.restore_from_path = MOLMIM_CHECKPOINTS_PATH
        model = MolMIMInference(cfg, interactive=True)

        return model

    def encode(self, smiles_list, use_batching=True, batch_size=1000):
        """
        Encode SMILES strings using the MolMIM model with optional batching.
        """
        smiles_list = self.to_list(smiles_list)

        if use_batching and len(smiles_list) > batch_size:
            all_latent_vectors = []
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i : i + batch_size]
                batch_latents = []
                for smiles in batch_smiles:
                    latent = self.model.seq_to_embeddings([smiles])
                    batch_latents.append(latent)
                all_latent_vectors.append(torch.cat(batch_latents))
            return torch.cat(all_latent_vectors)
        else:
            all_latent_vectors = []
            for smiles in smiles_list:
                latent = self.model.seq_to_embeddings([smiles])
                all_latent_vectors.append(latent)
            return torch.cat(all_latent_vectors)

    def decode(self, latent_vectors, use_batching=True, batch_size=1000):
        """
        Decode latent vectors back to SMILES using the MolMIM model with optional batching.
        """
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], 1, 512)
        if use_batching and len(latent_vectors) > batch_size:
            all_smiles = []
            for i in range(0, len(latent_vectors), batch_size):
                batch_latents = latent_vectors[i : i + batch_size]
                batch_enc = torch.ones((batch_latents.shape[0], 1), dtype=torch.bool).to(
                    self.model.device
                )
                batch_smiles = self.model.hiddens_to_seq(batch_latents, batch_enc)
                all_smiles.extend(batch_smiles)
            return np.array(all_smiles)
        else:
            enc_mask = torch.ones((latent_vectors.shape[0], 1), dtype=torch.bool).to(
                self.model.device
            )
            return np.array(self.model.hiddens_to_seq(latent_vectors, enc_mask))

    def get_latent_size(self):
        """
        Return the latent size of the MolMIM model.
        """
        return 512  # MolMIM latent size is fixed at 512