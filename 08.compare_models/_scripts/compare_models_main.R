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
# library(R.matlab)
library(loo)

load('_data/rlnc_data.RData')
sz <- dim(rlnc)
nSubjects <- sz[1]
nTrials   <- sz[2]

dataList <- list(nSubjects=nSubjects,
                 nTrials=nTrials, 
                 choice=rlnc[,,1], 
                 reward=rlnc[,,2])

# =============================================================================
#### Running Stan #### 
# =============================================================================
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

modelFile1 <- '_scripts/comparing_models_model1.stan'  # simple RL model
modelFile2 <- '_scripts/comparing_models_model2.stan'  # fictitious RL model

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

# =============================================================================
#### extract log_likelihood and compare models #### 
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


