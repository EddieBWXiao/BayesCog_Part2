# =============================================================================
#### Info #### 
# =============================================================================
# originally:
# (C) Dr. Lei Zhang, ALPN Lab, University of Birmingham
# l.zhang.13@bham.ac.uk

# B.X. modifications: 
# this document may accidentally become my new workflow...

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

# load('_data/rlnc_data.RData') #rlnc: a 10x100x2 thing
# sz <- dim(rlnc)
# nSubjects <- sz[1] #simulated 10 subjects
# nTrials   <- sz[2] #each had 100 trials
# 
# dataList <- list(nSubjects=nSubjects,
#                  nTrials=nTrials, 
#                  choice=rlnc[,,1], #third dimension 1st line choice, 1 or 2
#                  reward=rlnc[,,2]) #third dimension 2nd line outcome (reward), +1 or -1

# =============================================================================
# Simulations with my own combination of true parameters...
# =============================================================================
# # first, reverse-enginner the reward settings used in the .RData
# task1.c<-dataList$choice[201:300]
# task1.rew<-dataList$reward[201:300]
# task1.rew[task1.c==2] = -task1.rew[task1.c==2]
#     #(these were not the same for all ptp; this set of task looks nice)
# #then, throw these into the task
# task = list(outcome = data.frame(ref = task1.rew, alt = -task1.rew))
# plot(task$outcome$ref)
# #make the task longer...
# task = list(outcome = data.frame(ref = c(task1.rew,-task1.rew), alt = c(-task1.rew,task1.rew)))
# plot(task$outcome$ref) #this should sum to zero with task$outcome$alt
# save(task,file="_data/fixed_task.RData")

# after commenting out the above...
load("_data/fixed_task.RData") #this loads the variable, task
source('_scripts/RW1lr1beta_basic.R')
source('_scripts/RW1lr1beta_fict.R')
model2sim<-RW1lr1beta_fict #this is the model to generate the data

set.seed(1011)
nSubjects <- 40 #we decide this now
nTrials   <- dim(task$outcome)[1]
# lr.sim.raw <- rnorm(nSubjects, mean=0.6, sd=1)
# tau.sim.raw <- rnorm(nSubjects, mean=1.5, sd=0.3)
# lr.sim<-pnorm(lr.sim.raw)
# tau.sim<-pnorm(tau.sim.raw)*3
# note: decided instead to just generate an arbitrary range
lr.sim<-runif(nSubjects,min=0.2,max=0.9)
tau.sim<-runif(nSubjects,min=0.5,max=3)
hist(lr.sim)
hist(tau.sim)

#preallocate before actually generating:
dataList <- list(nSubjects=nSubjects,
                 nTrials=nTrials,
                 choice=array(NA,dim=c(nSubjects,nTrials)), 
                 reward=array(NA,dim=c(nSubjects,nTrials))) 
for(i in 1:nSubjects){
  sim<-RW1lr1beta_fict(c(lr.sim[i],tau.sim[i]),task)

  dataList$choice[i,]<-sim$choice
  dataList$reward[i,]<-sim$o_sim
  
  rm(sim)
}


# =============================================================================
#### Running Stan #### 
# =============================================================================
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

modelFile1 <- '_scripts/comparing_models_model1_ppc.stan'  # simple RL model, with ppc, S-by-T loglik
modelFile2 <- '_scripts/comparing_models_model2_ppc.stan'  # fictitious RL model, with ppc, S-by-T loglik

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
source('_scripts/HDIofMCMC.R')
#in this version, true parameters defined myself

fitted.lr.indv<-rstan::extract(fit_winner, pars = 'lr')$lr
fitted.lr.indv.mean<-colMeans(fitted.lr.indv)
fitted.lr.indv.hdi <- apply(fitted.lr.indv, 2, HDIofMCMC) #bounds
fitted.tau.indv<-rstan::extract(fit_winner, pars = 'tau')$tau
fitted.tau.indv.mean<-colMeans(fitted.tau.indv)
fitted.tau.indv.hdi <- apply(fitted.tau.indv, 2, HDIofMCMC) #lower bound


plot(fitted.lr.indv.mean,lr.sim)+abline(coef = c(0,1))
plot(fitted.tau.indv.mean,tau.sim)+abline(coef = c(0,1))

#more beautiful plots:
recov = data.frame(lr.fit = fitted.lr.indv.mean,
                   lr.sim = lr.sim,
                   lr.fit.l = fitted.lr.indv.hdi[1,], #lower bound
                   lr.fit.h = fitted.lr.indv.hdi[2,])
g.recov <- ggplot(recov,aes(lr.sim,lr.fit))+
    geom_point(alpha=0.5)+
    geom_pointrange(aes(ymin=lr.fit.l, ymax=lr.fit.h),alpha=0.5)+
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    theme_classic()
plot(g.recov)
# =============================================================================
#### Following this, I will do the model validation plots here
# =============================================================================

# trial-by-trial sequence; this is the "real data"
y_trial_mean = colMeans(dataList$choice == 1) #proportion of choosing 1, for each ptp
plot(1:nTrials, y_trial_mean,type='b') #quick check?

#extract trial-by-trial predicted choice (4000 arrays of 1 & 2, repeated for each ptp each trial)
y_pred_winner = extract(fit_winner, pars='y_pred')$y_pred
y_pred_winner_mean_mcmc = apply(y_pred_winner==1, c(1,3), mean) 
y_pred_winner_mean = colMeans(y_pred_winner_mean_mcmc) #mean choice on each trial
y_pred_winner_mean_HDI = apply(y_pred_winner_mean_mcmc, 2, HDIofMCMC)

#extract each participant's lose-shift rate
calc_lose_shift<-function(choices,outcomes,loseCode=0){
    shift <- diff(choices)!=0
    islose <- outcomes==loseCode
    islose<-islose[1:length(islose)-1] #get rid of end
    lsrate = mean(shift[islose])
    return(lsrate)   
}
calc_shift_rate<-function(choices){
    shift <- diff(choices)!=0
    rate = mean(shift)
    return(rate)   
}
#alright I did not see this coming: need to regenerate the outcomes... let's do the shift rate first
y_pred_winner_shiftrate = apply(y_pred_winner, c(1,2), calc_shift_rate) 
y_pred_winner_shiftrate_mean = colMeans(y_pred_winner_shiftrate)
data_shift_rate = apply(dataList$choice, 1, calc_shift_rate)
plot(data_shift_rate,y_pred_winner_shiftrate_mean)
#now try get the outcomes

#combine to get data for plotting
df = data.frame(Trial = 1:nTrials,
                Data  = y_trial_mean,
                model_mean = y_pred_winner_mean,
                HDI_l = y_pred_winner_mean_HDI[1,],
                HDI_h = y_pred_winner_mean_HDI[2,])

#PLOTTING
g1<-ggplot(df,aes(Trial,Data))+
    geom_line(aes(color='Data'))+ #the data
    geom_ribbon(aes(ymin= HDI_l, ymax= HDI_h, fill='model'), linetype=2, alpha=0.3)
g1 <- g1 + theme_classic() + scale_fill_manual(name = '',  values=c("model" = "indianred3")) +
    scale_color_manual(name = '',  values=c("Data" = "skyblue"))  +
    labs(y = 'Choosing ref (%)')
plot(g1)

tt_y = mean(df$Data)
df2 = data.frame(xx = c(rowMeans(y_pred_winner_mean_mcmc)) ,
                 model = rep(c('model'),each=4000) ) # overall mean, 4000 mcmc samples
g2 = ggplot(data=df2, aes(xx)) + 
    geom_histogram(data=subset(df2, model == 'model'),fill = "indianred3", alpha = 0.5, binwidth =.005)
g2 = g2 + geom_vline(xintercept=tt_y, color = 'skyblue2',size=1.5)
g2 = g2 + labs(x = 'Choosing ref (%)', y = 'Frequency')
g2 = g2 + theme_classic()
g2 = g2 + theme(axis.text   = element_text(size=22),
                axis.title  = element_text(size=25),
                legend.text = element_text(size=25))
plot(g2) #doesn't seem super-informative; since task volatile, near 50-50
