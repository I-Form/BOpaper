########################################################################################################################
'''BoTorch (max)'''

import numpy
import torch
data=numpy.genfromtxt('MultiObj.csv', delimiter=',', skip_header=1)
X_init=torch.from_numpy(data[:,0:4])
Y_init=(data[:,4:8]-data[:,4:8].mean(axis=0))/data[:,4:8].std(axis=0)
Y_init[:,3]=-Y_init[:,3]
Y_init=torch.from_numpy(Y_init)
parameter_space=torch.tensor([[8.0, 80.0, 0.5, 8.0], [15.0, 350.0, 1.5, 25.0]])


from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
Model=SingleTaskGP(train_X=X_init, train_Y=Y_init)
Model_mll=ExactMarginalLogLikelihood(Model.likelihood, Model)
fit_gpytorch_model(Model_mll)


from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition import qExpectedImprovement
from botorch.optim.optimize import optimize_acqf_list
batch_size=2
torch.manual_seed(seed=0)
AcqFun_list=[]
for _ in range(batch_size):
    weights=sample_simplex(Y_init.size()[1]).squeeze()
    objective=GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=Y_init))
    qEI_sample=SobolQMCNormalSampler(num_samples=2048, resample=False, seed=0)
    AcqFun=qExpectedImprovement(Model, best_f=objective(Y_init).max(), sampler=qEI_sample, 
                                objective=objective)
    AcqFun_list.append(AcqFun)
x_next, _ =optimize_acqf_list(AcqFun_list, bounds=parameter_space, num_restarts=50, 
                              raw_samples=512, options={"batch_limit": 10, "maxiter": 500})










########################################################################################################################
'''Trieste (min)'''

import numpy
import tensorflow as tf
from trieste.data import Dataset
from trieste.space import Box
data=numpy.genfromtxt('MultiObj.csv', delimiter=',', skip_header=1)
low_bnd=numpy.array([8, 80, 0.5, 8])
up_bnd=numpy.array([15, 350, 1.5, 25])
X_init=(data[:,0:4]-low_bnd)/(up_bnd-low_bnd)
Y_init=(data[:,4:8]-data[:,4:8].mean(axis=0))/data[:,4:8].std(axis=0)
Y_init[:,0:3]=-Y_init[:,0:3]
ExpData=Dataset(tf.constant(X_init), tf.constant(Y_init))
parameter_space=Box([0, 0, 0, 0], [1, 1, 1, 1])

import gpflow
from trieste.models.gpflow import GPflowModelConfig
from trieste.models import create_model
from trieste.models.model_interfaces import ModelStack
def build_model(ExpData, num_obj)->ModelStack:
    gprs=[] 
    kern=gpflow.kernels.Matern52()
    for iobj in range(num_obj): 
        single_data=Dataset(ExpData.query_points, tf.gather(ExpData.observations, [iobj], axis=1))
        gpr=gpflow.models.GPR(single_data.astuple(), kernel=kern, noise_variance=0.01)
        gpflow.set_trainable(gpr.likelihood, False)
        gprs.append((create_model(GPflowModelConfig(**{
            "model": gpr, 
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {"minimize_args": {"options": dict(maxiter=1000)}}
            })), 1))
    return ModelStack(*gprs)
numpy.random.seed(0)
tf.random.set_seed(0)
Model=build_model(ExpData, num_obj=4)

from trieste.acquisition.function import ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
BO=EfficientGlobalOptimization(builder=ExpectedHypervolumeImprovement(), num_query_points=1)
x_next=BO.acquire_single(search_space=parameter_space, dataset=ExpData, model=Model)
x_next.numpy()*(up_bnd-low_bnd)+low_bnd








