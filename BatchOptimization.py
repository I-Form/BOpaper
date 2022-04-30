########################################################################################################################
'''GPyOpt'''

import numpy
data=numpy.genfromtxt('BatchObj.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2,3]]
Y_init=(data[:,4]-data[:,4].mean())/data[:,4].std()
Y_init.shape=(27, 1)
parameter_space=[{'name': 'Saturation', 'type': 'continuous', 'domain': (35, 100)},
                 {'name': 'Layer_thickness', 'type': 'continuous', 'domain': (80, 120)},
                 {'name': 'Roll_speed', 'type': 'continuous', 'domain': (6, 14)},
                 {'name': 'Feed_powder_ratio', 'type': 'continuous', 'domain': (1, 3)}]

import GPyOpt
numpy.random.seed(123)
BO=GPyOpt.methods.BayesianOptimization(f=None, domain=parameter_space, X=X_init, 
                                       Y=Y_init, normalize_Y=False, 
                                       evaluator_type='thompson_sampling', 
                                       batch_size=2, maximize=True)
x_next=BO.suggest_next_locations()





























########################################################################################################################
'''Emukit (min)'''

import numpy
from emukit.core import ParameterSpace, ContinuousParameter
data=numpy.genfromtxt('BatchObj.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2,3]]
Y_init=-(data[:,4]-data[:,4].mean())/data[:,4].std()
Y_init.shape=(27, 1)
parameter_space=ParameterSpace([ContinuousParameter('Saturation', 35, 100), 
                                ContinuousParameter('Layer_thickness', 80, 120), 
                                ContinuousParameter('Roll_speed', 6, 14), 
                                ContinuousParameter('Feed_powder_ratio', 1, 3)])

import GPy
from emukit.model_wrappers import GPyModelWrapper
numpy.random.seed(0)
kern=GPy.kern.Matern52(input_dim=4, ARD=False)+GPy.kern.Bias(input_dim=4)
Model_gpy=GPy.models.GPRegression(X=X_init, Y=Y_init, kernel=kern, 
                                  normalizer=False, noise_var=0.01, mean_function=None)
Model_gpy.optimize()
Model_emukit=GPyModelWrapper(Model_gpy)

from emukit.bayesian_optimization.acquisitions.expected_improvement import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.optimization import GradientAcquisitionOptimizer
AcqFun=ExpectedImprovement(model=Model_emukit)
AcqFun_opt=GradientAcquisitionOptimizer(space=parameter_space)
BO=BayesianOptimizationLoop(space=parameter_space, model=Model_emukit, acquisition=AcqFun, 
                            batch_size=2, acquisition_optimizer=AcqFun_opt)
x_next=BO.get_next_points(None)


'''ExperimentalDesignLoop'''
from emukit.bayesian_optimization.acquisitions.local_penalization import LocalPenalization
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.core.optimization import GradientAcquisitionOptimizer
AcqFun=LocalPenalization(model=Model_emukit)
AcqFun_opt=GradientAcquisitionOptimizer(space=parameter_space)
BO=ExperimentalDesignLoop(space=parameter_space, model=Model_emukit, acquisition=AcqFun, 
                          batch_size=2, acquisition_optimizer=AcqFun_opt)
x_next=BO.get_next_points(None)







########################################################################################################################
'''Dragonfly (max)'''

import numpy
from dragonfly.exd import domains
data=numpy.genfromtxt('BatchObj.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2,3]]
Y_init=(data[:,4]-data[:,4].mean())/data[:,4].std()
parameter_space=domains.EuclideanDomain([[35, 100], [80, 120], [6, 14], [1, 3]])

from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.opt import gp_bandit
func_caller=EuclideanFunctionCaller(None, parameter_space)
BO=gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
numpy.random.seed(0)
BO.initialise()
for i in range(0, 27):
    BO.tell([(X_init[i], Y_init[i])])
x_next=BO.ask(n_points=2)






























########################################################################################################################
'''BoTorch (max)'''

import numpy
import torch
data=numpy.genfromtxt('BatchObj.csv', delimiter=',', skip_header=1)
X_init=torch.from_numpy(data[:,[0,1,2,3]])
Y_init=numpy.array((data[:,4]-data[:,4].mean())/data[:,4].std())
Y_init.shape=(27, 1)
Y_init=torch.from_numpy(Y_init)
parameter_space=torch.tensor([[35.0, 80.0, 6.0, 1.0], [100.0, 120.0, 14.0, 3.0]])

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
Model=SingleTaskGP(train_X=X_init, train_Y=Y_init)
Model_mll=ExactMarginalLogLikelihood(Model.likelihood, Model)
fit_gpytorch_model(Model_mll)

from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
qEI_sample=SobolQMCNormalSampler(num_samples=2048, resample=False, seed=0)
AcqFun=qExpectedImprovement(Model, best_f=Y_init.max(), sampler=qEI_sample)
torch.manual_seed(seed=0)
x_next, _=optimize_acqf(acq_function=AcqFun, bounds=parameter_space, q=2, 
                        num_restarts=50, raw_samples=512, sequential=False)






















########################################################################################################################
'''Trieste (min)'''

import numpy
import tensorflow as tf
import trieste
from trieste.space import Box
data=numpy.genfromtxt('BatchObj.csv', delimiter=',', skip_header=1)
low_bnd=numpy.array([35, 80, 6, 1])
up_bnd=numpy.array([100, 120, 14, 3])
X_init=(data[:,[0,1,2,3]]-low_bnd)/(up_bnd-low_bnd)
Y_init=-(data[:,4]-data[:,4].mean())/data[:,4].std()
Y_init.shape=(27, 1)
ExpData=trieste.data.Dataset(tf.constant(X_init), tf.constant(Y_init))
parameter_space=Box([0, 0, 0, 0], [1, 1, 1, 1])

import gpflow
from trieste.models.gpflow import GPflowModelConfig
from trieste.models import create_model
def build_model(data):
    kern=gpflow.kernels.Matern52()
    gpr=gpflow.models.GPR(data=data.astuple(), kernel=kern, noise_variance=0.01)
    gpflow.set_trainable(gpr.likelihood, False)
    model_spec={
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {"minimize_args": {"options": dict(maxiter=1000)}},
        }
    return GPflowModelConfig(**model_spec)
numpy.random.seed(0)
tf.random.set_seed(0)
Model=create_model(build_model(ExpData))

from trieste.acquisition import BatchMonteCarloExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
AcqFun=BatchMonteCarloExpectedImprovement(sample_size=1000)
BO=EfficientGlobalOptimization(builder=AcqFun, num_query_points=2)
x_next=BO.acquire_single(search_space=parameter_space, dataset=ExpData, model=Model)
x_next.numpy()*(up_bnd-low_bnd)+low_bnd






