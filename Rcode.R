#----------------------------------------------------------------------------------------------------------------------------------#
# batch optimization #
#----------------------------------------------------------------------------------------------------------------------------------#
data<-read.csv("BatchObj.csv", header=TRUE)
data[ ,5]<-(data[ ,5]-mean(data[ ,5]))/sd(data[ ,5])
#--------------------------------------------------------------------------------------------#
library("DiceOptim")
set.seed(0)
fitted.model<-km(~1, design=data.frame(data[ ,1:4]), response=data[ ,5], covtype="matern5_2", 
                 optim.method="BFGS", multistart=100, control=list(trace=FALSE, pop.size=50))
BO<-max_qEI(fitted.model, npoints=2, lower=c(35, 80, 6, 1),  upper=c(100, 120, 14, 3), 
            crit="exact", minimization=FALSE, optimcontrol=list(nStarts=10, method="BFGS"))
x_next=BO$par
#--------------------------------------------------------------------------------------------#
library("mlrMBO")
parameter_space=makeParamSet(makeNumericParam("Saturation", lower=35, upper=100), 
                             makeNumericParam("Layer_thickness", lower=80, upper=120), 
                             makeNumericParam("Roll_speed", lower=6, upper=14), 
                             makeNumericParam("Feed_powder_ratio", lower=1, upper=3))
ctrl=makeMBOControl(propose.points=2, final.method="best.predicted", store.model.at=1)
ctrl=setMBOControlInfill(ctrl, filter.proposed.points=TRUE)
ctrl=setMBOControlMultiPoint(ctrl, method="moimbo", moimbo.objective="mean.se.dist", 
                             moimbo.dist="nearest.better", moimbo.maxit=500L)
set.seed(0)
BO<-initSMBO(par.set=parameter_space, design=data.frame(lapply(data, as.numeric)), 
             control=ctrl, minimize=FALSE, noisy=TRUE)
x_next=proposePoints(BO)$prop.points







#----------------------------------------------------------------------------------------------------------------------------------#
# multi-objective #
#----------------------------------------------------------------------------------------------------------------------------------#
data<-read.csv("MultiObj.csv", header=TRUE)
data[ ,5]<-(data[ ,5]-mean(data[ ,5]))/sd(data[ ,5])
data[ ,6]<-(data[ ,6]-mean(data[ ,6]))/sd(data[ ,6])
data[ ,7]<-(data[ ,7]-mean(data[ ,7]))/sd(data[ ,7])
data[ ,8]<-(data[ ,8]-mean(data[ ,8]))/sd(data[ ,8])

library("mlrMBO")
parameter_space=makeParamSet(makeNumericParam("Binder_amount", lower=8, upper=15), 
                             makeNumericParam("Particle_size", lower=80, upper=350), 
                             makeNumericParam("Nozzle_diameter", lower=0.5, upper=1.5),
                             makeNumericParam("Printing_speed", lower=8, upper=25))
ctrl=makeMBOControl(n.objectives=4, propose.points=2, store.model.at=1)
ctrl=setMBOControlInfill(ctrl, crit=makeMBOInfillCritEI(), filter.proposed.points=TRUE)
ctrl=setMBOControlMultiPoint(ctrl, method="cl")
ctrl=setMBOControlMultiObj(ctrl, method="parego")
set.seed(0)
BO<-initSMBO(par.set=parameter_space, design=data.frame(lapply(data, as.numeric)), 
             control=ctrl, minimize=c(FALSE, FALSE, FALSE, TRUE), noisy=FALSE)
x_next=proposePoints(BO)$prop.points














#----------------------------------------------------------------------------------------------------------------------------------#
# BB constraint #
#----------------------------------------------------------------------------------------------------------------------------------#
data<-read.csv("C:/Users/Administrator/OneDrive - TCDUD.onmicrosoft.com/Acdemic/ResearchDocuments/13-SequencialDoE/example/BBcon.csv", header=TRUE)
data[ ,4]<--(data[ ,4]-mean(data[ ,4]))/sd(data[ ,4])
data[ ,5]<-data[ ,5]-10

library("DiceOptim")
set.seed(0)
fitted.model<-km(~1, design=data.frame(data[ ,1:3]), response=data[ ,4], covtype="matern5_2", 
                 nugget=0.001, nugget.estim=TRUE, optim.method="BFGS", multistart=1000, 
                 control=list(trace=FALSE, pop.size=500))
fitted.cst<-km(~1, design=data.frame(data[ ,1:3]), response=data[ ,5], covtype="matern5_2", 
               nugget=0.001, nugget.estim=TRUE, optim.method="BFGS", multistart=1000, 
               control=list(trace=FALSE, pop.size=500))
BO<-critcst_optimizer(crit="EFI", fitted.model, fitted.cst, equality=FALSE, lower=c(100, 100, 80), 
                      upper=c(300, 1500, 200), type="UK", 
                      critcontrol=list(tolConstraints=0.001, n.mc=1000, slack=TRUE),
                      optimcontrol=list(method="genoud", max.generations=100, pop.size=200))
x_next=BO$par



