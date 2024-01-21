# =============================================================================
#### Info #### 
# =============================================================================
# true generating model: fictitious model
#
# true parameters: lr  = rnorm(10, mean=0.6, sd=0.12); tau = rnorm(10, mean=1.5, sd=0.2)
#
# (C) Dr. Lei Zhang, ALPN Lab, University of Birmingham
# l.zhang.13@bham.ac.uk


# =============================================================================
#### Construct Data #### 
# =============================================================================
# clear workspace
rm(list = ls())
library(rstan)
library(ggplot2)
library(R.matlab)
library(loo)

#     tmp <- readMat('_data/rlnc_data.mat')
#     rlnc <- tmp$data
#     save(rlnc, file = "_data/rlnc_data.RData")

load('_data/rlnc_data.RData') #rlnc: a 10x100x2 thing
sz <- dim(rlnc)
nSubjects <- sz[1] #simulated 10 subjects
nTrials   <- sz[2] #each had 100 trials

dataList <- list(nSubjects=nSubjects,
                 nTrials=nTrials, 
                 choice=rlnc[,,1], #third dimension 1st line choice, 1 or 2
                 reward=rlnc[,,2]) #third dimension 2nd line outcome (reward), +1 or -1

# =============================================================================
#### Running Stan #### 
# =============================================================================
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

modelFile1 <- '_scripts/comparing_models_model1.stan'  # simple RL model
modelFile2 <- '_scripts/comparing_models_model2.stan'  # fictitious RL model
modelFile3 <- '_scripts/comparing_models_model1_ppc.stan'  # simple RL model, with ppc, S-by-T loglik
modelFile4 <- '_scripts/comparing_models_model2_ppc.stan'  # fictitious RL model, with ppc, S-by-T loglik

nIter     <- 2000
nChains   <- 4 
nWarmup   <- floor(nIter/2)
nThin     <- 1

### model1
cat("Estimating", modelFile1, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rl1 <- stan(modelFile1, 
               data    = dataList, 
               chains  = nChains,
               iter    = nIter,
               warmup  = nWarmup,
               thin    = nThin,
               init    = "random",
               seed    = 145015634
)

cat("Finishing", modelFile1, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")


### model2
cat("Estimating", modelFile2, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rl2 <- stan(modelFile2, 
                data    = dataList, 
                chains  = nChains,
                iter    = nIter,
                warmup  = nWarmup,
                thin    = nThin,
                init    = "random",
                seed    = 1450154637
)

cat("Finishing", modelFile2, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")


### model3
cat("Estimating", modelFile3, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rl3 <- stan(modelFile3, 
                data    = dataList, 
                chains  = nChains,
                iter    = nIter,
                warmup  = nWarmup,
                thin    = nThin,
                init    = "random",
                seed    = 1450154637
)

cat("Finishing", modelFile3, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")


### model4 --> problem: I don't think the file exists
# cat("Estimating", modelFile4, "model... \n")
# startTime = Sys.time(); print(startTime)
# cat("Calling", nChains, "simulations in Stan... \n")
# 
# fit_rl4 <- stan(modelFile4, 
#                 data    = dataList, 
#                 chains  = nChains,
#                 iter    = nIter,
#                 warmup  = nWarmup,
#                 thin    = nThin,
#                 init    = "random",
#                 seed    = 1450154637
# )
# 
# cat("Finishing", modelFile4, "model simulation ... \n")
# endTime = Sys.time(); print(endTime)  
# cat("It took",as.character.Date(endTime - startTime), "\n")

# =============================================================================
#### extract log_likelihood and compare models #### 
# ? wait, is the _master script not the most up-to-date one?
# =============================================================================
LL1 <- extract_log_lik(fit_rl1)
LL2 <- extract_log_lik(fit_rl2)

rel_n_eff1 = loo::relative_eff(exp(LL1), chain_id = rep(1:nChains, each = nIter - nWarmup))
rel_n_eff2 = loo::relative_eff(exp(LL2), chain_id = rep(1:nChains, each = nIter - nWarmup))

loo1 <- loo(LL1, r_eff = rel_n_eff1)
loo2 <- loo(LL2, r_eff = rel_n_eff2)
loo::loo_compare(loo1, loo2) # positive difference indicates the 2nd model's predictive accuracy is higher


loo_list = list(loo1, loo2)
model_weights = loo::loo_model_weights(loo_list, method = 'stacking', optim_control = list(reltol=1e-10))
# loo_model_weights(loo_list, method = 'pseudobma') - optional


## Use bridge sampling
library(bridgesampling)
bridge_rl1 <- bridge_sampler(samples = fit_rl1)
bridge_rl2 <- bridge_sampler(samples = fit_rl2)
BF_bridge <- bf(bridge_rl2, bridge_rl1)
print(BF_bridge)
#OKAY for some reason, the Bayes Factor is extremely large
log(BF_bridge$bf) #this number is suspiciously similar to the elpd_diff?!

# =============================================================================
#### visualise participant parameters
#copying from 06.'s reinforcement_learning_multi_parm_main_master.R
# =============================================================================
#since 2 is the winner...
fit_winner <-fit_rl2

#the learning rate
plot_dens_lr  <- stan_plot(fit_winner, pars=c('lr'), show_density=T, fill_color = 'skyblue')
plot_dens_lr <- plot_dens_lr +xlim(0,1) #overwrite the range
print(plot_dens_lr)
#inverse temperature
plot_dens_tau  <- stan_plot(fit_winner, pars=c('tau'), show_density=T, fill_color = 'skyblue')
print(plot_dens_tau)

#look at group-level parameter; note that these were in... logit space? Difficult to compare with "true" simulated group mean
plot_dens_grp <- stan_plot(fit_winner, pars=c('lr_mu_raw','lr_sd_raw', 'tau_mu_raw','tau_sd_raw'), show_density=T, fill_color = 'skyblue')
print(plot_dens_grp)

# =============================================================================
#### "parameter recovery"
# =============================================================================
#I got this from the .mat in _data
true.params<-read.csv("_data/rlnc_parms.csv") #working directory is 08.compare_models
fitted.lr.indv<-rstan::extract(fit_winner, pars = 'lr')$lr
fitted.lr.indv.mean<-colMeans(fitted.lr.indv)
fitted.tau.indv<-rstan::extract(fit_winner, pars = 'tau')$tau
fitted.tau.indv.mean<-colMeans(fitted.tau.indv)
plot(fitted.lr.indv.mean,true.params$lr)+abline(coef = c(0,1))
plot(fitted.tau.indv.mean,true.params$tau)+abline(coef = c(0,1))
plot(fitted.lr.indv.mean,true.params$lr[10:1])+abline(coef = c(0,1))
plot(fitted.tau.indv.mean,true.params$tau[10:1])+abline(coef = c(0,1))
#issues from first run: not aligned; reversed vector direction and still not great?

#?? I hope these lr are the actual LR, not lr_raw (which would be the deviation of each subject from mean, scaled by the lr_sd_raw?)

