import numpy as np
data=np.genfromtxt('SurfRough.csv', delimiter=',', skip_header=1)
X_init=data[:,[0,1,2]]
Y_init=data[:,3]
Preference=[] #create the preferential data P
for i in range(0, X_init.shape[0]):
    for j in range(i+1, X_init.shape[0]):
        if Y_init[j]-Y_init[i]>=2:
            Preference.append([i,j])
        elif Y_init[j]-Y_init[i]<=-2:
            Preference.append([j,i])
maX=X_init[np.argmax(Y_init)] #best experiment 
Class=list(np.ones(X_init.shape[0])) #all experiments have outputs
parameter_space=[[180, 240], [20, 35], [500, 800]]

import GPy, commoncode, BO
#function calls to experiment at X
def query(X):
    return np.array([1.0])
#function returns 0 if X is an invalid setting (experiment no output), 1 otherwise
def valid(X, f):
    return np.array([1.0])
#RBF kernel function
def RBF(X1, X2, hype, diag_=False):
    lengthscale=hype['lengthscale']['value']
    variance=hype['variance']['value']
    kernel=GPy.kern.RBF(X1.shape[1], lengthscale=lengthscale, variance=variance, ARD=True)
    return kernel.K(X1,X2)
kernel=RBF #initialize kernel
#initial values for the hyperparameters of the kernel
hype0={'lengthscale':{'value':np.ones((1, X_init.shape[1]))*0.3,
                      'range':np.vstack([[np.exp(-5.0), np.exp(5.0)]]*X_init.shape[1]),
                      'transform': commoncode.logexp()},
       'variance':{'value':np.array([10.0]),
                   'range':np.vstack([[np.exp(-5.0), np.exp(4.1)]]),
                   'transform': commoncode.logexp()},
       'noise_variance':{'value':np.array([1.0]),
                         'range':np.vstack([[1.0, 1.0001]]),
                         'transform': commoncode.logexp()}}
PBO=BO.BO(X_init, Preference, Class, parameter_space, kernel, hype0, query, valid, 
          maX, surrogateM='SGP', acquisition='UCB')
#update the kernel hyperparameters and compute the next point
PBO.find_next(query, 0, update_model='True')
print("next point = ", PBO.X[-1,:])