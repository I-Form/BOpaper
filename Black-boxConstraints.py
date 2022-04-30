import numpy
import torch
data=numpy.genfromtxt('BBcon.csv', delimiter=',', skip_header=1)
X_init=torch.from_numpy(data[:,[0,1,2]])
Y_init=data[:,[3,4]]
Y_init[:,0]=(Y_init[:,0]-Y_init[:,0].mean())/Y_init[:,0].std()
Y_init[:,1]=Y_init[:,1]-10
Y_init=torch.from_numpy(Y_init)
parameter_space=torch.tensor([[100.0, 100.0, 80.0], [300.0, 1500.0, 200.0]])

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
Model=SingleTaskGP(train_X=X_init, train_Y=Y_init)
Model_mll=ExactMarginalLogLikelihood(Model.likelihood, Model)
fit_gpytorch_model(Model_mll)

from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
def obj_callable(Z):
    return Z[..., 0]
def constraint_callable(Z):
    return Z[..., 1]
constrained_obj=ConstrainedMCObjective(objective=obj_callable, 
                                       constraints=[constraint_callable])
qEI_sample=SobolQMCNormalSampler(num_samples=2048, resample=False, seed=0)
AcqFun=qExpectedImprovement(Model, best_f=(Y_init[:,0]*(Y_init[:,1]<=0)).max(), 
                            sampler=qEI_sample, objective=constrained_obj)
torch.manual_seed(seed=0)
x_next, _=optimize_acqf(acq_function=AcqFun, bounds=parameter_space, q=2, 
                        num_restarts=50, raw_samples=512, sequential=False)















########################################################################################################################
'''following the website'''
import numpy
import torch
data=numpy.genfromtxt('BBcon.csv', delimiter=',', skip_header=1)
X_init=torch.from_numpy(data[:,[0,1,2]])
Y_obj=(data[:,3]-data[:,3].mean())/data[:,3].std()
Y_obj.shape=(17, 1)
Y_obj=torch.from_numpy(Y_obj)
Y_con=data[:,4]-10
Y_con.shape=(17, 1)
Y_con=torch.from_numpy(Y_con)
parameter_space=torch.tensor([[100.0, 100.0, 80.0], [300.0, 1500.0, 200.0]])

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
Model_obj=SingleTaskGP(train_X=X_init, train_Y=Y_obj)
Model_obj_mll=ExactMarginalLogLikelihood(Model_obj.likelihood, Model_obj)
fit_gpytorch_model(Model_obj_mll)
Model_con=SingleTaskGP(train_X=X_init, train_Y=Y_con)
Model_con_mll=ExactMarginalLogLikelihood(Model_con.likelihood, Model_con)
fit_gpytorch_model(Model_con_mll)

from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.models import ModelListGP
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
def obj_callable(Z):
    return Z[..., 0]
def constraint_callable(Z):
    return Z[..., 1]
constrained_obj=ConstrainedMCObjective(objective=obj_callable, 
                                       constraints=[constraint_callable])
qEI_sample=SobolQMCNormalSampler(num_samples=2048, resample=False, seed=0)
AcqFun=qExpectedImprovement(ModelListGP(Model_obj, Model_con), best_f=(Y_obj*(Y_con<=0)).max(), 
                            sampler=qEI_sample, objective=constrained_obj)
torch.manual_seed(seed=0)
x_next, _=optimize_acqf(acq_function=AcqFun, bounds=parameter_space, q=2, 
                        num_restarts=50, raw_samples=512, sequential=False)