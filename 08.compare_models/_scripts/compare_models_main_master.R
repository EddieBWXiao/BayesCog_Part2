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
cat("Estimating", modelFile4, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rl4 <- stan(modelFile4,
                data    = dataList,
                chains  = nChains,
                iter    = nIter,
                warmup  = nWarmup,
                thin    = nThin,
                init    = "random",
                seed    = 1450154637
)

cat("Finishing", modelFile4, "model simulation ... \n")
endTime = Sys.time(); print(endTime)
cat("It took",as.character.Date(endTime - startTime), "\n")

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

#note: for the _ppc stan, because log_lik changed... can it still work? yes!
LL3 <- extract_log_lik(fit_rl3)
LL4 <- extract_log_lik(fit_rl4)
rel_n_eff3 = loo::relative_eff(exp(LL3), chain_id = rep(1:nChains, each = nIter - nWarmup))
rel_n_eff4 = loo::relative_eff(exp(LL4), chain_id = rep(1:nChains, each = nIter - nWarmup))
loo3 <- loo(LL3, r_eff = rel_n_eff3)
loo4 <- loo(LL4, r_eff = rel_n_eff4)
loo::loo_compare(loo3, loo4)

#repeat bridge sampling
bridge_rl3 <- bridge_sampler(samples = fit_rl3)
bridge_rl4 <- bridge_sampler(samples = fit_rl4)
BF_bridge_check <- bf(bridge_rl4, bridge_rl3)
print(BF_bridge_check)

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

# =============================================================================
### Posterior predictive check (using generated quantities)
#### migrated from the other script, comparing models ppc
# basically, this works when the two ppc scripts work; comments added by B.X.
# not just one model but has two models plotted side-by-side for comparison
# =============================================================================

source('_scripts/HDIofMCMC.R')
#should be able to move this whole thing into another script...
f_rl<-fit_rl3
f_rlfic<-fit_rl4

# overall mean
mean(dataList$choice[1:2,] == 1 ) #uh, why is this just selecting ptp1 & 2?

# trial-by-trial sequence; this is the "real data"
y_mean = colMeans(dataList$choice == 1) #proportion of choosing 1, for each ptp
plot(1:100, y_mean,type='b') #quick check?

#extract trial-by-trial predicted choice (4000 arrays of 1 & 2, repeated for each ptp each trial)
y_pred_rl = extract(f_rl, pars='y_pred')$y_pred
y_pred_rlfic = extract(f_rlfic, pars='y_pred')$y_pred

y_pred_rl_mean_mcmc = apply(y_pred_rl==1, c(1,3), mean) 
y_pred_rl_mean = colMeans(y_pred_rl_mean_mcmc)
y_pred_rl_mean_HDI = apply(y_pred_rl_mean_mcmc, 2, HDIofMCMC)

y_pred_rlfic_mean_mcmc = apply(y_pred_rlfic==1, c(1,3), mean)
y_pred_rlfic_mean = colMeans(y_pred_rlfic_mean_mcmc)
y_pred_rlfic_mean_HDI = apply(y_pred_rlfic_mean_mcmc, 2, HDIofMCMC)

#more explanation on the above code:
    #second input to apply() specifies, counterintuitively, the dimensions we do NOT apply the function over?
    #this way, for each dim1 (MCMC sample) and each dim3 (trial), we have a mean predicted choice
    #i.e., we averaged over participants
    #the y_pred_rl_mean further gets one mean per column (colMeans is NOT mean over column to get row measn!!!)
    #which is like the "mean estimate" (and the MCMC samples form the confidence interval)

#plot(1:100, colMeans(y_pred_mean),type='b')

# =============================================================================
#### make plots #### 
# =============================================================================
myconfig <- theme_bw(base_size = 20) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank() )

df = data.frame(Trial = 1:100,
                Data  = y_mean,
                RL = y_pred_rl_mean,
                RL_HDI_l = y_pred_rl_mean_HDI[1,],
                RL_HDI_h = y_pred_rl_mean_HDI[2,],
                RLfic = y_pred_rlfic_mean,
                RLfic_HDI_l = y_pred_rlfic_mean_HDI[1,],
                RLfic_HDI_h = y_pred_rlfic_mean_HDI[2,])

## time course of the choice
g1 = ggplot(df, aes(Trial,Data))
g1 = g1 + geom_line(size = 1.5, aes(color= 'Data')) + geom_point(size = 2, shape = 21, fill='skyblue3',color= 'skyblue3')
g1 = g1 + geom_ribbon(aes(ymin=RL_HDI_l, ymax=RL_HDI_h, fill='RL'), linetype=2, alpha=0.3)
g1 = g1 + geom_ribbon(aes(ymin=RLfic_HDI_l, ymax=RLfic_HDI_h, fill='RLfic'), linetype=2, alpha=0.3)
g1 = g1 + myconfig + scale_fill_manual(name = '',  values=c("RL" = "skyblue3", "RLfic" = "indianred3")) +
    scale_color_manual(name = '',  values=c("Data" = "skyblue"))  +
    labs(y = 'Choosing correct (%)')
g1 = g1 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
g1
ggsave(plot = g1, "_plots/compare_choice_seq_ppc.png", width = 8, height = 4, type = "cairo-png", units = "in")


## "at participant-level": correlation between model agnostic metrics for real data and simulated
#(Still under construction)

## overall choice: true data (vertical line) + model prediction (hist)
tt_y = mean(df$Data)
df2 = data.frame(xx = c(rowMeans(y_pred_rl_mean_mcmc),rowMeans(y_pred_rlfic_mean_mcmc)) ,
                 model = rep(c('RL','RLfic'),each=4000) ) # overall mean, 4000 mcmc samples

g2 = ggplot(data=df2, aes(xx)) + 
    geom_histogram(data=subset(df2, model == 'RL'),fill = "skyblue3", alpha = 0.5, binwidth =.005) +
    geom_histogram(data=subset(df2, model == 'RLfic'),fill = "indianred3", alpha = 0.5, binwidth =.005)
g2 = g2 + geom_vline(xintercept=tt_y, color = 'skyblue2',size=1.5)
g2 = g2 + labs(x = 'Choosing correct (%)', y = 'Frequency')
g2 = g2 + myconfig# + scale_x_continuous(breaks=c(tt_y), labels=c("Event1")) 
g2 = g2 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
g2
ggsave(plot = g2, "_plots/compare_choice_mean_ppc.png", width = 6, height = 4, type = "cairo-png", units = "in")



