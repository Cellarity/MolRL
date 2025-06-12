from rdkit import Chem
from rdkit import RDLogger
import os

def valid(smi: str):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def get_vae_model(model_file_name='2024-12-20-vae-rnn-pubchem-epoch=81.ckpt',device='cuda:0'):
    from molrl.rnn import SMILESVAERNNCAT
    package_dir = os.path.dirname(os.path.abspath(__file__))
    model_checkpoints_path = os.path.join(package_dir, 'model_checkpoints')
    model_filename = model_file_name
    model_path = os.path.join(model_checkpoints_path, model_filename)
    print('Loading model from:', model_path)
    model = SMILESVAERNNCAT.load_from_checkpoint(model_path, map_location=device)
    model = model.eval()
    return model