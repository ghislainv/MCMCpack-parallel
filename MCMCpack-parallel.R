##=========================================================
##
## Running MCMCpack in parallel with doParallel and foreach
##
## Ghislain Vieilledent <ghislain.vieilledent@cirad.fr>
## Septemeber 2016
##
##=========================================================

## Load libraries
library(MCMCpack)
library(doParallel)
library(foreach)
library(ggplot2)
library(gridExtra)

## Create directories to hold results
dir.create("results")
dir.create("figs")

##==============================================================
##
## Generating data for a Hierarchical Gaussian Linear Regression
##
##==============================================================

# Constants
nobs <- 1000
nspecies <- 20
species <- c(1:nspecies,sample(c(1:nspecies),(nobs-nspecies),replace=TRUE))

# Covariates
X1 <- runif(n=nobs,min=0,max=10)
X2 <- runif(n=nobs,min=0,max=10)
X <- cbind(rep(1,nobs),X1,X2)
W <- X

# Target parameters
# beta
beta.target <- matrix(c(0.1,0.3,0.2),ncol=1)
# Vb
Vb.target <- c(0.5,0.2,0.1)
# b
b.target <- cbind(rnorm(nspecies,mean=0,sd=sqrt(Vb.target[1])),
                  rnorm(nspecies,mean=0,sd=sqrt(Vb.target[2])),
                  rnorm(nspecies,mean=0,sd=sqrt(Vb.target[3])))
# sigma2
sigma2.target <- 0.25

# Response
Y <- vector()
for (n in 1:nobs) {
  Y[n] <- rnorm(n=1,
                mean=X[n,]%*%beta.target+W[n,]%*%b.target[species[n],],
                sd=sqrt(sigma2.target))
}

# Data-set
Data <- as.data.frame(cbind(Y,X1,X2,species))
Data$species <- factor(Data$species)

# Observation plot
g1 <- ggplot(data=Data,aes(x=X1,y=Y)) + geom_point(aes(colour=species))
g2 <- ggplot(data=Data,aes(x=X2,y=Y)) + geom_point(aes(colour=species))
plot.obs <- grid.arrange(g1,g2,ncol=2)
ggsave(filename="figs/observations.png",plot=plot.obs,width=30,height=15,unit=c("cm"))

##===============================================================
##
## Setting up the cluster and starting values for each MCMC chain
##
##===============================================================

## Make a cluster for parallel MCMCs
nchains <- 2
ncores <- nchains ## One core for each MCMC chains
clust <- makeCluster(ncores)
registerDoParallel(clust)

## Starting values and random seed
seed <- 1234
set.seed(seed)
beta.start <- runif(nchains,-1,1)
sigma2.start <- runif(nchains,1,10)
Vb.start <- runif(nchains,1,10)
seed.mcmc <- round(runif(nchains,0,1e6))

##==============================================================
##
## Estimating parameter posterior distribution with MCMChregress
##
##==============================================================

mod.MCMChregress <- foreach (i=1:nchains, .packages="MCMCpack") %dopar% {
  mod <- MCMChregress(fixed=Y~X1+X2, random=~X1+X2, group="species",
                      data=Data, burnin=1000, mcmc=1000, thin=1,verbose=1,
                      seed=seed.mcmc[i], beta.start=beta.start[i], sigma2.start=sigma2.start[i],
                      Vb.start=Vb.start[i], mubeta=0, Vbeta=1.0E6,
                      r=3, R=diag(c(1,0.1,0.1)), nu=0.001, delta=0.001)
  return(mod)
}

## Stop cluster
stopCluster(clust)

## Extract list of MCMCs from output
mod.mcmc <- mcmc.list(lapply(mod.MCMChregress,"[[","mcmc"))

## Outputs summary
mod.stat <- summary(mod.mcmc)$statistics
sink(file="results/mcmc_summary.txt")
mod.stat
sink()

## Plot trace and posterior distributions
pdf("figs/mcmc_trace.pdf")
plot(mod.mcmc)
dev.off()

## Predictive posterior mean for each observation
str(mod.MCMChregress[[1]])
Y.pred <- mod.MCMChregress[[1]]$Y.pred

## Predicted-Observed
pdf("figs/pred_obs.pdf")
plot(Data$Y,Y.pred)
abline(a=0,b=1,col="red")
dev.off()

##===========================================================================
## End of script
##===========================================================================
