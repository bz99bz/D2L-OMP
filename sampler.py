import os
import pickle
import torch
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader_energy import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
    load_sampling_fn
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
from rdkit.Chem import AllChem, rdMolDescriptors


# -------- Sampler for molecule generation tasks --------
class Sampler_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)

        self.configt = self.ckpt_dict['config']

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'],
                                              self.device)

        self.model_x_fra = load_model_from_ckpt(self.ckpt_dict['params_x_fra'], self.ckpt_dict['x_state_dict_fra'],
                                               self.device)
        self.model_adj_fra = load_model_from_ckpt(self.ckpt_dict['params_adj_fra'],
                                                  self.ckpt_dict['adj_state_dict_fra'],
                                                  self.device)
        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)  # for init_flags
        with open(f'data/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD

        self.init_flags = init_flags(self.train_graph_list, self.configt, 10000).to(self.device[0])

        x, adj, _ = self.sampling_fn(self.model_x, self.model_adj,self.model_x_fra, self.model_adj_fra, self.init_flags)

        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)

        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5

        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
            for smiles in gen_smiles:
                f.write(f'{smiles}\n')

        # -------- Evaluation --------
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles,
                                 train=train_smiles)
        scores_nspdk = eval_graph_list(self.test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

        logger.log(f'Number of molecules: {num_mols}')
        logger.log(f'validity w/o correction: {num_mols_wo_correction / num_mols}')
        for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']:
            logger.log(f'{metric}: {scores[metric]}')
        logger.log(f'NSPDK MMD: {scores_nspdk}')
        logger.log('=' * 100)

