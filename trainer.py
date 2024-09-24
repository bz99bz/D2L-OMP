import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from utils.loader_faster import load_seed, load_device, load_data, load_data_fra,load_model_params, load_model_optimizer,load_ema, load_loss_fn, load_batch,load_sampling_fn
from utils.logger import Logger, set_log, start_log, train_log
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx,correct_mol, valid_mol,construct_mol
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol

from min_norm_solvers import MinNormSolver, gradient_normalizers
# from moses.metrics.metrics import get_all_metrics
from parsers.config import get_config
from evaluation.stats import eval_graph_list
from sampler_faster import *

from rdkit.Chem.Descriptors import qed
from rdkit import Chem
import utils.environment as env
from utils.sascorer import readFragmentScores



#multi_task_loss
def loss_fn_multi(gen_mols,t,_fscores):
    # print("_fscores:",_fscores)
    fn_start = time.time()
    qed_test=[]
    logp_test=[]

    # -------- Evaluation --------
    valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in gen_mols]
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]

    for mol in valid_mols:
        if(len(Chem.MolToSmiles(mol))):
            if(t=='1'):
               qed_test.append(qed(mol))
            if(t=='2'):
               logp_test.append(env.penalized_logp(mol,_fscores))
            # logp_test.append(Chem.Descriptors.MolLogP(mol))
    qed_mean = 0
    if (len(qed_test)):
        qed_mean = np.mean(qed_test)
    qed_max = 0
    if (len(qed_test)):
        qed_max = max(qed_test)
    logp_mean = 0
    if (len(logp_test)):
        logp_mean = np.mean(logp_test)
    logp_max = 0
    if (len(logp_test)):
        logp_max = max(logp_test)
    if (t == '1'):
        print("time:", time.time() - fn_start)
        print("qed_max", qed_max)
        mse = nn.MSELoss()
        qed_loss = mse(torch.tensor(qed_mean), torch.tensor(1.0))
        print("qed_loss", qed_loss)
        return 10*qed_loss
    if (t == '2'):
        print("time:", time.time() - fn_start)
        print("logp_max", logp_max)
        mse = nn.MSELoss()
        logp_loss = mse(torch.tensor(logp_mean), torch.tensor(10.0))
        print("logp_loss", logp_loss)
        return 0.0001 * logp_loss

class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)
        self.train_fra_loader, self.test_fra_loader = load_data_fra(self.config)


        self.params_x, self.params_adj = load_model_params(self.config)
        self.params_x_fra, self.params_adj_fra = load_model_params(self.config)

    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.params_x, self.config.train,
                                                                                self.device)
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj,
                                                                                      self.config.train,
                                                                                      self.device)

        self.model_x_fra, self.optimizer_x_fra, self.scheduler_x_fra = load_model_optimizer(self.params_x_fra, self.config.train,
                                                                                self.device)
        self.model_adj_fra, self.optimizer_adj_fra, self.scheduler_adj_fra = load_model_optimizer(self.params_adj_fra,
                                                                                      self.config.train,
                                                                                      self.device)


        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)
        self.ema_x_fra = load_ema(self.model_x_fra, decay=self.config.train.ema)
        self.ema_adj_fra = load_ema(self.model_adj_fra, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)
        configt = get_config(self.config.name.name, self.seed)
        config = get_config(f"sample_{self.config.name.name}", self.seed)
        train_graph_list, test_graph_list = load_data(configt, get_graph_list=True)
        sampling_fn = load_sampling_fn(configt, config.sampler, config.sample, self.device)
        init_flags_train = init_flags(train_graph_list, configt, 1000).to(self.device[0])
        _fscores=readFragmentScores(name='fpscores')
        # -------- Training --------
        x_test = 0
        adj_test = 0
        for epoch in trange(0, (self.config.train.num_epochs), desc='[Epoch]', position=1, leave=False):

            self.train_x = []
            self.train_adj = []
            self.train_x_fra = []
            self.train_adj_fra = []
            self.test_x = []
            self.test_adj = []
            self.test_x_fra = []
            self.test_adj_fra = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()
            self.model_x_fra.train()
            self.model_adj_fra.train()

            n_iter = 0

            for _, train_b in enumerate(self.train_loader):
                n_iter += 1
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()


                x, adj = load_batch(train_b, self.device)
                if (epoch < 2) or ((n_iter-1) % 100 > 0):
                    loss_subject = (x, adj)
                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                else:
                    ######Multi-task part#######
                    print("epoch:",epoch)
                    tasks = ['1', '2']
                    loss_data_x = {}
                    loss_data_adj = {}
                    scale_x = {}
                    scale_adj = {}
                    grads_x = {}
                    grads_adj = {}
                    task_loss_x = {}
                    task_loss_adj = {}
                    losses_vec_x = []
                    losses_vec_adj = []

                    ##add Multi-task-based loss


                    ###train faster###
                    # First compute representations (z)
                    # with torch.no_grad():
                    #       mols_volatile = Variable(mols_b.data)
                    loss_subject_variable = (x, adj)
                    loss_x_rep, loss_adj_rep = self.loss_fn(self.model_x, self.model_adj, *loss_subject_variable)


                    # As an approximate solution we only need gradients for input
                    if isinstance(loss_x_rep, list):
                        loss_x_rep=loss_x_rep[0]
                        loss_x_rep_variable = [Variable(loss_x_rep.data.clone(), requires_grad=True)]
                        list_x_rep = True

                    else:
                        loss_x_rep_variable = Variable(loss_x_rep.data.clone(), requires_grad=True)
                        list_x_rep = False

                    if isinstance(loss_adj_rep, list):
                        loss_adj_rep=loss_adj_rep[0]
                        loss_adj_rep_variable = [Variable(loss_adj_rep.data.clone(), requires_grad=True)]
                        list_adj_rep = True
                    else:
                        loss_adj_rep_variable = Variable(loss_adj_rep.data.clone(), requires_grad=True)
                        list_adj_rep = False

                    if (n_iter-1) % 100 == 0:
                        x_save=x_test
                        adj_save=adj_test
                        print("n_iter-1:",n_iter-1)
                        x_test, adj_test, _ = sampling_fn(self.model_x, self.model_adj, init_flags_train)
                        samples_int = quantize_mol(adj_test)
                        samples_int = samples_int - 1
                        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
                        print("samples_int",samples_int)
                        try:
                           adj_test = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3,1, 2)
                        except RuntimeError:
                            x_test=x_save
                            adj_test=adj_save
                        x_test = torch.where(x_test > 0.5, 1, 0)
                        x_test = torch.concat([x_test, 1 - x_test.sum(dim=-1, keepdim=True)],
                                              dim=-1)  # 32, 9, 4 -> 32, 9, 5
                    gen_mols, num_mols_wo_correction = gen_mol(x_test, adj_test, configt.data.data)
                    for t in tasks:
                        self.optimizer_x.zero_grad()
                        self.optimizer_adj.zero_grad()


                        # Comptue gradients of each loss function wrt parameters
                        # Reasoning based on shared parameters has done above

                        task_loss_adj[t]=task_loss_x[t] = loss_fn_multi(gen_mols, t,_fscores)


                        print("task_loss_x[t]", task_loss_x[t])
                        print("task_loss_adj[t]", task_loss_adj[t])
                        task_loss_total_x = loss_x_rep_variable + task_loss_x[t]
                        task_loss_total_adj = loss_adj_rep_variable + task_loss_adj[t]
                        print("task_loss_total_x", task_loss_total_x)
                        print("task_loss_total_adj", task_loss_total_adj)
                        losses_vec_x.append(task_loss_total_x)
                        losses_vec_adj.append(task_loss_total_adj)

                        # Original loss + single indicator  MSE loss
                        loss_data_x[t] = task_loss_total_x.item()
                        loss_data_adj[t] = task_loss_total_adj.item()

                        loss_x = task_loss_total_x
                        loss_adj = task_loss_total_adj
                        loss_x.backward(retain_graph=True)
                        loss_adj.backward(retain_graph=True)
                        grads_x[t] = []
                        grads_adj[t] = []

                        if list_x_rep:
                            grads_x[t].append(Variable(loss_x_rep_variable[0].grad.data.clone(), requires_grad=False))
                            loss_x_rep_variable[0].grad.data.zero_()

                        else:
                            grads_x[t].append(Variable(loss_x_rep_variable.grad.data.clone(), requires_grad=False))
                            loss_x_rep_variable.grad.data.zero_()

                        if list_adj_rep:
                            grads_adj[t].append(Variable(loss_adj_rep_variable[0].grad.data.clone(), requires_grad=False))
                            loss_adj_rep_variable[0].grad.data.zero_()
                        else:
                            grads_adj[t].append(Variable(loss_adj_rep_variable.grad.data.clone(), requires_grad=False))
                            loss_adj_rep_variable.grad.data.zero_()

                    gn_x = gradient_normalizers(grads_x, loss_data_x, "loss+")
                    gn_adj = gradient_normalizers(grads_adj, loss_data_adj, "loss+")

                    for t in tasks:
                        for gr_i in range(len(grads_x[t])):
                            grads_x[t][gr_i] = grads_x[t][gr_i] / gn_x[t]

                    for t in tasks:
                        for gr_i in range(len(grads_adj[t])):
                            grads_adj[t][gr_i] = grads_adj[t][gr_i] / gn_adj[t]

                    # Frank-Wolfe iteration to compute scales.

                    sol_x, min_norm_x = MinNormSolver.find_min_norm_element([grads_x[t] for t in tasks])
                    sol_adj, min_norm_adj = MinNormSolver.find_min_norm_element([grads_adj[t] for t in tasks])

                    for i, t in enumerate(tasks):
                        scale_x[t] = float(sol_x[i])
                        scale_adj[t] = float(sol_adj[i])

                    self.optimizer_x.zero_grad()
                    self.optimizer_adj.zero_grad()
                    if (n_iter - 1) % 100 == 0:
                        print("n_iter-1:", n_iter - 1)
                        x_test, adj_test, _ = sampling_fn(self.model_x, self.model_adj, init_flags_train)
                        samples_int = quantize_mol(adj_test)

                        samples_int = samples_int - 1
                        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

                        adj_test = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3,
                                                                                                                 1, 2)
                        x_test = torch.where(x_test > 0.5, 1, 0)
                        x_test = torch.concat([x_test, 1 - x_test.sum(dim=-1, keepdim=True)], dim=-1)
                    for i, t in enumerate(tasks):
                        multi_loss=loss_fn_multi(gen_mols, t,_fscores)
                        loss_t_x = loss_x + multi_loss
                        loss_t_adj = loss_adj + multi_loss
                        loss_data_x[t] = loss_t_x.item()
                        loss_data_adj[t] = loss_t_adj.item()
                        if i > 0:
                            task_loss_total_x = task_loss_total_x + scale_x[t] * loss_t_x
                            task_loss_total_adj = task_loss_total_adj + scale_adj[t] * loss_t_adj
                        else:
                            task_loss_total_x = scale_x[t] * loss_x
                            task_loss_total_adj = scale_adj[t] * loss_adj
                    # task_loss_total_x.backward()
                    # task_loss_total_adj.backward()
                    loss_x = task_loss_total_x
                    loss_adj = task_loss_total_adj

                loss_x.backward()
                loss_adj.backward()

                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj.parameters(), self.config.train.grad_norm)

                self.optimizer_x.step()
                self.optimizer_adj.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()

            self.model_x.eval()
            self.model_adj.eval()



            for _, train_b_fra in enumerate(self.train_fra_loader):
                self.optimizer_x_fra.zero_grad()
                self.optimizer_adj_fra.zero_grad()
                x_fra, adj_fra = load_batch(train_b_fra, self.device)
                loss_subject_fra = (x_fra, adj_fra)
                loss_x_fra, loss_adj_fra = self.loss_fn(self.model_x_fra, self.model_adj_fra, *loss_subject_fra)
                loss_x_fra.backward()
                loss_adj_fra.backward()
                torch.nn.utils.clip_grad_norm_(self.model_x_fra.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj_fra.parameters(), self.config.train.grad_norm)
                self.optimizer_x_fra.step()
                self.optimizer_adj_fra.step()
                self.ema_x_fra.update(self.model_x_fra.parameters())
                self.ema_adj_fra.update(self.model_adj_fra.parameters())
                self.train_x_fra.append(loss_x_fra.item())
                self.train_adj_fra.append(loss_adj_fra.item())

            if self.config.train.lr_schedule:
                self.scheduler_x_fra.step()
                self.scheduler_adj_fra.step()

            self.model_x_fra.eval()
            self.model_adj_fra.eval()


            for _, test_b in enumerate(self.test_loader):
                x, adj = load_batch(test_b, self.device)
                loss_subject = (x, adj)

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            for _, test_b_fra in enumerate(self.test_fra_loader):
                x_fra, adj_fra = load_batch(test_b_fra, self.device)
                loss_subject_fra = (x_fra, adj_fra)

                with torch.no_grad():
                    self.ema_x_fra.store(self.model_x_fra.parameters())
                    self.ema_x_fra.copy_to(self.model_x_fra.parameters())
                    self.ema_adj_fra.store(self.model_adj_fra.parameters())
                    self.ema_adj_fra.copy_to(self.model_adj_fra.parameters())

                    loss_x_fra, loss_adj_fra = self.loss_fn(self.model_x_fra, self.model_adj_fra, *loss_subject_fra)
                    self.test_x_fra.append(loss_x_fra.item())
                    self.test_adj_fra.append(loss_adj_fra.item())

                    self.ema_x_fra.restore(self.model_x_fra.parameters())
                    self.ema_adj_fra.restore(self.model_adj_fra.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)
            mean_train_x_fra = np.mean(self.train_x_fra)
            mean_train_adj_fra = np.mean(self.train_adj_fra)
            mean_test_x_fra = np.mean(self.test_x_fra)
            mean_test_adj_fra = np.mean(self.test_adj_fra)

            # -------- Log losses --------
            logger.log(f'{epoch + 1:03d} | {time.time() - t_start:.2f}s | '
                       f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                       f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | '
                       f'test x_fra: {mean_test_x_fra:.3e} | test adj_fra: {mean_test_adj_fra:.3e} | '
                       f'train x_fra: {mean_train_x_fra:.3e} | train adj_fra: {mean_train_adj_fra:.3e} | '
                       , verbose=False)

            # -------- Sav       e checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
                save_name = f'_{epoch + 1}' if epoch < self.config.train.num_epochs - 1 else ''

                torch.save({
                    'model_config': self.config,
                    'params_x': self.params_x,
                    'params_adj': self.params_adj,
                    'params_x_fra': self.params_x_fra,
                    'params_adj_fra': self.params_adj_fra,
                    'x_state_dict': self.model_x.state_dict(),
                    'adj_state_dict': self.model_adj.state_dict(),
                    'x_state_dict_fra': self.model_x_fra.state_dict(),
                    'adj_state_dict_fra': self.model_adj_fra.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                    'ema_adj': self.ema_adj.state_dict(),
                    'ema_x_fra': self.ema_x_fra.state_dict(),
                    'ema_adj_fra': self.ema_adj_fra.state_dict()
                }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')

            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                tqdm.write(f'[EPOCH {epoch + 1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | '
                           f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e} | test adj_fra: {mean_test_adj_fra:.3e} | train adj_fra: {mean_train_adj_fra:.3e} | '
                           f'test x_fra: {mean_test_x_fra:.3e} | train x_fra: {mean_train_x_fra:.3e}')
        print(' ')
        return self.ckpt
