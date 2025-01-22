from logging import raiseExceptions
import torch
import math
import numpy as np
from functorch import vmap, jacrev, jacfwd
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import copy
from diffuse.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
    LossType, ModelMeanType, ModelVarType
from diffuse.nn import DiffAtt
from diffuse.utils import full_DAG
import random

class Diffuse():
    def __init__(self, small_layer,n_nodes,  beta_start, beta_end,
                 epochs, batch_size, n_steps,learning_rate: float = 0.001):

        self.n_nodes = n_nodes
        assert self.n_nodes > 1, "Not enough nodes, make sure the dataset contain at least 2 variables (columns)."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Diffusion parameters
        self.n_steps = n_steps
        betas = get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=self.n_steps, scale=1,
                                        beta_start=beta_start, beta_end=beta_end)
        self.gaussian_diffusion = GaussianDiffusion(betas=betas,
                                                    loss_type=LossType.MSE,
                                                    model_mean_type=ModelMeanType.EPSILON,  # START_X,EPSILON
                                                    model_var_type=ModelVarType.FIXED_LARGE,
                                                    rescale_timesteps=True,
                                                    )


        self.schedule_sampler = UniformSampler(self.gaussian_diffusion)

        ## Diffusion training
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = DiffAtt(n_nodes,small_layer).to(self.device)
        self.model.float()
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.val_diffusion_loss = []
        self.best_loss = float("inf")
        self.early_stopping_wait = 300

        ## Topological Ordering

        self.masking = True
        self.residue = False

    def fit1(self, X,condition):
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)
        X = torch.FloatTensor(X).to(self.device)

        condition = (condition - condition.mean(0, keepdims=True)) / condition.std(0, keepdims=True)
        self.condition = torch.FloatTensor(condition).to(self.device)

        self.train_score(X)

    def fit2(self, prunX):
        prunX = (prunX - prunX.mean(0, keepdims=True)) / prunX.std(0, keepdims=True)
        prunX = torch.FloatTensor(prunX).to(self.device)
        if prunX.shape[1]<=5:
            self.finetuning(prunX)
        order = self.topological_ordering(prunX)
        dag_order=np.array(full_DAG(order))

        out_dag = self.pruning_by_coef_2nd(dag_order, prunX.detach().cpu().numpy())

        return out_dag


    def format_seconds(self,seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


    def pruning_by_coef_2nd(self, graph_batch, X) -> np.ndarray:
        """
        for a given graph, pruning the edge according to edge weights;
        quadratic regression for each causal regression for edge weights and then
        thresholding
        对于给定的图，根据边的权值对边进行修剪，对每条边进行二次回归，对边的权值进行因果回归，然后进行阈值分割
        """

        start_time = time.time()
        thresh = 0.18
        d = graph_batch.shape[0]
        reg = LinearRegression()

        poly = PolynomialFeatures()
        W = []
        W_continuous = []

        pbar = tqdm(range(d), desc="Pruning_by_coef_2nd")
        for i in pbar:
            col = graph_batch[i] > 0.1

            if np.sum(col) <= 0.1:
                W.append(np.zeros(d))
                W_continuous.append(np.zeros(d))
                continue

            X_train = X[:, col]
            X_train_expand = poly.fit_transform(X_train)[:, 1:]
            X_train_expand_names = poly.get_feature_names()[1:]

            y = X[:, i]
            reg.fit(X_train_expand, y)
            reg_coeff = reg.coef_
            reg_coeff = torch.from_numpy(reg_coeff)
            # reg_coeff = torch.softmax(reg_coeff, dim=-1)
            cj = 0
            new_reg_coeff = np.zeros(d, )
            new_reg_coeff_continuous = np.zeros(d, )
            for ci in range(d):
                if col[ci]:
                    xxi = 'x{}'.format(cj)
                    for iii, xxx in enumerate(X_train_expand_names):
                        if xxi in xxx:

                            if reg_coeff[iii] > thresh:
                                new_reg_coeff[ci] = 1
                                new_reg_coeff_continuous[ci] = reg_coeff[iii]

                            break
                    cj += 1
            W.append(new_reg_coeff)
            W_continuous.append(new_reg_coeff_continuous)

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        formatted_time = self.format_seconds(total_elapsed_time)
        print("prun_time: ", formatted_time)

        return np.array(W)


    def train_score(self, X):
        start_time = time.time()
        best_model_state_epoch = 300
        self.model.train()
        n_samples = X.shape[0]

        val_ratio = 0.3
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size

        X_train, X_val = X[:train_size], X[train_size:]
        data_loader_val = torch.utils.data.DataLoader(X_val, min(val_size, self.batch_size))
        data_loader = torch.utils.data.DataLoader(X_train, min(train_size, self.batch_size), drop_last=True)
        data_loader_condition = torch.utils.data.DataLoader(self.condition, min(train_size, self.batch_size), drop_last=True)
        pbar = tqdm(range(self.epochs), desc="Training Epoch")


        for epoch in pbar:
            loss_per_step = []

            for steps, (x_start, condition) in enumerate(zip(data_loader, data_loader_condition)):

                # apply noising and masking
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                noise = torch.randn_like(x_start).to(self.device)
                x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)

                model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t),
                                          condition, self.gaussian_diffusion._scale_timesteps(t))


                diffusion_losses = (noise - model_output) ** 2
                diffusion_loss = (
                            diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                loss_per_step.append(diffusion_loss.item())
                self.opt.zero_grad()
                diffusion_loss.backward()
                self.opt.step()

            if epoch % 10 == 0 and epoch > best_model_state_epoch:
                with torch.no_grad():
                    loss_per_step_val = []
                    for steps, (x_start, condition) in enumerate(zip(data_loader_val, data_loader_condition)):
                        t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                        noise = torch.randn_like(x_start).to(self.device)
                        x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)

                        model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t),
                                                  condition, self.gaussian_diffusion._scale_timesteps(t))
                        diffusion_losses = (noise - model_output) ** 2
                        diffusion_loss = (diffusion_losses.mean(
                            dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                        loss_per_step_val.append(diffusion_loss.item())
                    epoch_val_loss = np.mean(loss_per_step_val)

                    if self.best_loss > epoch_val_loss:
                        self.best_loss = epoch_val_loss
                        self.best_model_state = deepcopy(self.model.state_dict())
                        best_model_state_epoch = epoch
                pbar.set_postfix({'Epoch Loss': epoch_val_loss})

            if epoch - best_model_state_epoch > self.early_stopping_wait:  # Early stopping
                break


        print(f"Early stoping at epoch {epoch}")
        print(f"Best model at epoch {best_model_state_epoch} with loss {self.best_loss}")


        end_time = time.time()
        total_elapsed_time = end_time - start_time
        formatted_time = self.format_seconds(total_elapsed_time)
        print("train_time: ",formatted_time)

    def finetuning(self, X):
        self.model.load_state_dict(self.best_model_state)

        self.model.train()
        pbar = tqdm(range(10), desc="finetuning Epoch")

        data_loader_condition = torch.utils.data.DataLoader(self.condition, self.batch_size)
        selected_condition = next(iter(data_loader_condition))

        for epoch in pbar:
            t, weights = self.schedule_sampler.sample(X.shape[0], self.device)
            noise = torch.randn_like(X).to(self.device)
            x_t = self.gaussian_diffusion.q_sample(X, t, noise=noise)

            model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t),
                                      selected_condition, self.gaussian_diffusion._scale_timesteps(t))

            diffusion_losses = (noise - model_output) ** 2
            diffusion_loss = (
                    diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()

            self.opt.zero_grad()
            diffusion_loss.backward()
            self.opt.step()

    def topological_ordering(self, X, step=None):
        start_time = time.time()
        topo_batch_size=50
        self.model.eval()
        order = []

        self.active_nodes = list(range(self.n_nodes))
        steps_list = [step] if step is not None else range(0, self.n_steps + 1, self.n_steps // 3)
        pbar = tqdm(range(len(self.active_nodes) - 1), desc="Nodes ordered ")

        for jac_step in pbar:
            leaves = []
            for i, steps in enumerate(steps_list):
                self.t_functorch = steps
                leaf_ = self.compute_jacobian_and_get_leaf(X)
                leaves.append(leaf_)
            leaf = Counter(leaves).most_common(1)[0][0]
            leaf_global = self.active_nodes[leaf]
            order.append(leaf_global)
            self.active_nodes.pop(leaf)

        order.append(self.active_nodes[0])
        order.reverse()

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        formatted_time = self.format_seconds(total_elapsed_time)
        print("topological_time: ", formatted_time)

        return order
 
 
    def model_fn_functorch(self,X):

        indices_np =  np.full(X.shape[0], self.t_functorch)
        indices = torch.from_numpy(indices_np).long().to(self.device)

        indices_np_condition =  np.full(self.condition.shape[0], self.t_functorch)
        indices_condition = torch.from_numpy(indices_np_condition).long().to(self.device)

        score_active = self.model(X, self.gaussian_diffusion._scale_timesteps(indices), self.condition,
                                  self.gaussian_diffusion._scale_timesteps(indices_condition))[:,self.active_nodes].squeeze()
        return score_active

    def get_masked(self, X):
        dropout_mask = torch.zeros_like(X).to(self.device)
        dropout_mask[:, self.active_nodes] = 1

        return (X * dropout_mask).float()


    def compute_jacobian_and_get_leaf(self, X):

        with torch.no_grad():

            epsilon = -0.2
            x_batch_dropped = self.get_masked(X) if self.masking else X
            outputs = self.model_fn_functorch(x_batch_dropped)

            i2 = 0
            jacobian = torch.zeros(len(self.active_nodes), len(self.active_nodes)).float().to(self.device)

            for i1 in range(self.n_nodes):

                if float(x_batch_dropped[0, i1]) != 0:
                    perturbed_inputs = x_batch_dropped.clone()
                    perturbed_inputs[:, i1] += epsilon
                    perturbed_outputs = self.model_fn_functorch(perturbed_inputs)
                    jac = (perturbed_outputs - outputs)[:, i2] / epsilon
                    variance = torch.var(jac).item()
                    jacobian[i2, i2] = variance
                    i2 += 1

            leaf = self.get_leaf(jacobian)

        return leaf

    def get_leaf(self, jacobian_active):

        jacobian_active = jacobian_active.cpu().numpy()
        jacobian_var_diag = jacobian_active.diagonal()
        var_sorted_nodes = np.argsort(jacobian_var_diag)

        leaf_current = var_sorted_nodes[0]
        return leaf_current