import torch
import numpy as np
import abc
from tqdm import trange

from losses import get_score_fn
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE,VESDE
from utils.graph_utils import quantize_mol
from utils.mol_utils import gen_mol
import torch.nn as nn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, sde_fra,score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.sde_fra=sde_fra
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, sde_fra,score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.sde_fra = sde_fra
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, obj, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if ("fra" in obj):
            self.obj = obj[:-4]
        else:
            self.obj = obj

    def update_fn(self, x, adj, flags, t):
        dt = -1. / self.rsde.N

        if self.obj == 'x':
            z = gen_noise(x, flags, sym=False)
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
            return x, x_mean

        elif self.obj == 'adj':
            z = gen_noise(adj, flags)
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
            adj_mean = adj + drift * dt
            adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, obj, sde,sde_fra, score_fn, probability_flow=False):
        super().__init__(sde,sde_fra, score_fn, probability_flow)
        if ("fra" in obj):
            self.obj = obj[:-4]
        else:
            self.obj = obj

    def update_fn(self, x, adj, x_fra,adj_fra, flags, t):

        if self.obj == 'x':
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
            z = gen_noise(x, flags, sym=False)
            z_fra = 0.01 * gen_noise(x_fra, flags, sym=False)
            x_mean = x - f
            x = x_mean + G[:, None, None] * (z+ z_fra)
            return x, x_mean

        elif self.obj == 'adj':
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
            z = gen_noise(adj, flags)
            z_fra = 0.01* gen_noise(adj_fra, flags, sym=False)
            adj_mean = adj - f
            adj = adj_mean + G[:, None, None] * (z+ z_fra)
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        if ("fra" in obj):
            self.obj = obj[:-4]
        else:
            self.obj = obj
        pass

    def update_fn(self, x, adj, flags, t):
        if self.obj == 'x':
            return x, x
        elif self.obj == 'adj':
            return adj, adj
        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
    def __init__(self, obj, sde,sde_fra, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde,sde_fra, score_fn, snr, scale_eps, n_steps)
        if ("fra" in obj):
            self.obj = obj[:-4]
        else:
            self.obj = obj

    def update_fn(self, x, adj,x_fra,adj_fra, flags, t,configt):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        if self.obj == 'x':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                noise = gen_noise(x, flags, sym=False)
                noise_fra = 0.01 * gen_noise(x_fra, flags, sym=False)
                if isinstance(sde, VESDE):
                   sigma = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t
                else:
                   sigma=sde.beta_0 + t * (sde.beta_1 - sde.beta_0)
                weight_t = sigma /alpha
                l = 0.05
                energy=0
                if(t[0]*10000%5==0):
                   energy = energy_guidence(x, adj, t, configt)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x_mean=x_mean+l*weight_t[:, None, None] * energy
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * (noise+noise_fra) * seps
            return x, x_mean

        elif self.obj == 'adj':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                noise = gen_noise(adj, flags)
                noise_fra = 0.75 * gen_noise(adj_fra, flags)
                sigma = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t
                weight_t = sigma / alpha
                l=0.05
                energy=0
                if (t[0] * 10000 % 5 == 0):
                   energy = energy_guidence(x, adj, t, configt)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad
                adj_mean = adj_mean + l*weight_t[:, None, None]* energy
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * (noise+noise_fra) * seps
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, sde_x_fra, sde_adj_fra,shape_x, shape_adj,shape_x_fra, shape_adj_fra, configt,predictor='Euler', corrector='None',
                   snr=0.1, scale_eps=1.0, n_steps=1,
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    def pc_sampler(model_x, model_adj,model_x_fra, model_adj_fra, init_flags):
        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)


        predictor_fn = ReverseDiffusionPredictor if predictor == 'Reverse' else EulerMaruyamaPredictor
        corrector_fn = LangevinCorrector if corrector == 'Langevin' else NoneCorrector

        predictor_obj_x = predictor_fn('x', sde_x,sde_x_fra, score_fn_x, probability_flow)
        corrector_obj_x = corrector_fn('x', sde_x, sde_x_fra,score_fn_x, snr, scale_eps, n_steps)


        predictor_obj_adj = predictor_fn('adj', sde_adj, sde_adj_fra,score_fn_adj, probability_flow)
        corrector_obj_adj = corrector_fn('adj', sde_adj, sde_adj_fra,score_fn_adj, snr, scale_eps, n_steps)


        with torch.no_grad():
            # -------- Initial sample --------
            x = sde_x.prior_sampling(shape_x).to(device)
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
            x_fra = sde_x_fra.prior_sampling(shape_x).to(device)
            adj_fra = sde_adj_fra.prior_sampling_sym(shape_adj).to(device)
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            x_fra = mask_x(x_fra, flags)
            adj_fra = mask_adjs(adj_fra, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

            # -------- Reverse diffusion process --------
            for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t


                _x = x
                _adj=adj
                x, x_mean = corrector_obj_x.update_fn(x, adj, x_fra,adj_fra, flags, vec_t,configt)
                adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, x_fra,adj_fra, flags,vec_t, configt)


                _x = x
                _adj = adj
                x, x_mean = predictor_obj_x.update_fn(x, adj, x_fra,adj_fra, flags, vec_t)
                adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, x_fra,adj_fra, flags, vec_t)


            print(' ')
            return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)

    return pc_sampler

def energy_guidence(x,adj,t,configt):
    # enegry-guidence

    qed_list = []
    samples_int = quantize_mol(adj)

    samples_int = samples_int - 1
    samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

    energy_adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)

    energy_x = torch.where(x > 0.5, 1, 0)
    energy_x = torch.concat([energy_x, 1 - energy_x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5

    gen_mols, num_mols_wo_correction = gen_mol(energy_x, energy_adj, configt.data.data)
    num_mols = len(gen_mols)
    mse = nn.MSELoss()
    validity = num_mols_wo_correction / num_mols
    energy_validity = mse(torch.tensor(validity), torch.tensor(1.0))
    energy = energy_validity
 
  

    return energy

# -------- S4 solver --------
def S4_solver(sde_x, sde_adj, sde_x_fra, sde_adj_fra,shape_x, shape_adj, shape_x_fra, shape_adj_fra,predictor='None', corrector='None',
              snr=0.1, scale_eps=1.0, n_steps=1,
              probability_flow=False, continuous=False,
              denoise=True, eps=1e-3, device='cuda'):
    def s4_solver(model_x, model_adj, model_x_fra, model_adj_fra,init_flags):

        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)
        # score_fn_x_fra = get_score_fn(sde_x_fra, model_x_fra, train=False, continuous=continuous)
        # score_fn_adj_fra = get_score_fn(sde_adj_fra, model_adj_fra, train=False, continuous=continuous)


        with torch.no_grad():
            # -------- Initial sample --------
            x = sde_x.prior_sampling(shape_x).to(device)
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
            dt = -1. / diff_steps

            # -------- Rverse diffusion process --------
            for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt / 2)

                # -------- Score computation --------
                score_x = score_fn_x(x, adj, flags, vec_t)
                score_adj = score_fn_adj(x, adj, flags, vec_t)

                Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
                Sdrift_adj = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

                # -------- Correction step --------
                timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                if isinstance(sde_x, VPSDE):
                    alpha = sde_x.alphas.to(vec_t.device)[timestep]
                else:
                    alpha = torch.ones_like(vec_t)

                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * score_x
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                if isinstance(sde_adj, VPSDE):
                    alpha = sde_adj.alphas.to(vec_t.device)[timestep]  # VP
                else:
                    alpha = torch.ones_like(vec_t)  # VE
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * score_adj
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

                # -------- Prediction step --------
                x_mean = x
                adj_mean = adj
                mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)
                x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                x = x + Sdrift_x * dt
                adj = adj + Sdrift_adj * dt

                mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
                x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                x_mean = mu_x
                adj_mean = mu_adj
            print(' ')
            return (x_mean if denoise else x), (adj_mean if denoise else adj), 0

    return s4_solver
