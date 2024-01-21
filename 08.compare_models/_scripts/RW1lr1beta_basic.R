RW1lr1beta_basic = function(params,task){
  # written in R by Tim Sandhu; adapted 2023.12.8 by Eddie Xiao
  # note this is heavily based on the Hanneke den Ouden tutorial
  # http://www.hannekedenouden.ruhosting.nl/RLtutorial/html/RL_topPage.html 
  # to confirm: if adapted version matches MATLAB for given data
  
  # task - list or data frame
    #task$outcome: vector or array
    #task$choice: 1 and 2 for index of the option (arm) chosen
  # params - c(alpha,beta), or list(alpha=__,beta=__); list helps offload need to memorise parameter names
  
  # parse input
  if(is.list(params)){
    # add input of parameters variables from parameters
    alpha<-params$alpha
    beta<-params$beta
  }else{
    alpha = params[1]
    beta = params[2]
  }
    
  if(is.data.frame(task)){
    n_trials = dim(task)[1] #row for number of trials
  }else{
    n_trials = dim(task$outcome)[1]
    if("choice"%in% names(task)){
      n_trials = length(task$choice)
    }
  }
  
  outcome = task$outcome
  
  #extract choices & see if in simulation mode
  if("choice" %in% names(task)){
    choice = task$choice
    sim_mode=FALSE
  }else{
    sim_mode = TRUE
    choice = rep(NA,n_trials)
  }
  
  # initialize variables
  v0 = c(.0,.0) #important: this should be the case if outcomes are 1 or -1
  value = array(NA,c(n_trials,2)) #to store value (for both arms)
  choice_prob = array(NA,c(n_trials,2))
  o_sim = rep(NA,n_trials)
  pchoice = rep(NA,n_trials) #to distinguish from choice_prob --> single p of making current choice
  loglik1 = 0 #a tally for log likelihood
  v = v0 # value to be updated
  
  # loop along trials
  for (t in 1:n_trials){
    # compute p(choice) based on value
    ev = exp(beta*v) # exponentiated value
    p = ev/sum(ev) # softmax; output a vector of p
    
    if(sim_mode){
      # weighted coin flip
      if (runif(1)<=p[1]){ #probablt never runif(1)=0.5 but...
        c = 1
      } else {
        c = 2
      }
      choice[t]=c
      o = outcome[t,c] #in sim mode, input outcome should have two columns; outcome received will be the chosen
    }else{
      # extract choice and outcome for this trial (if not doing simulations)
      c = choice[t]
      o = outcome[t] #in non-simulations, outcome is the outcome received
    }
    o_sim[t] = o
    
    # store value and choice prob 
    value[t,] = v #this is always v from the PREVIOUS trial
    choice_prob[t,] = p
    pchoice[t] = p[c] 
    
    # increment llh of observed choices
    loglik1 = loglik1 + log(p[c]) # c here for observed choice
    
    # compute prediction error based on outcome and update value
    pe = o - v[c] # pe
    v[c] = v[c] + alpha*pe # update value
  }#end of loop over trials
  
  # compute the likelihood: sum of logs of p(observed choice) for all trials
  # use this to find the maximum
  if(sim_mode){
    loglik = NA
  }else{
    loglik = sum(log(pchoice))
  }
  
  return(list('value'=value,
              "v_final"=v, #for the value learnt on final trial
              "PE_final" = pe,#for the prediction error on final trial
              'pchoice'=pchoice, 
              'choice'=choice,
              'choice_prob'=choice_prob,
              'o_sim' = o_sim,
              'loglik'=loglik,
              'loglik1'=loglik1))
}
