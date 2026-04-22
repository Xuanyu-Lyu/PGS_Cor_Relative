# ----------------------------------------------------------------------------------------------------------------------
# Program: oneACEvc.R
# Author: Hermine Maes
# Date: 10 22 2018
#
# Twin Univariate ACE model to estimate causes of variation across multiple groups
# Matrix style model - Raw data - Continuous data
# -------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
# Load Libraries & Options
rm(list=ls())
library(OpenMx)
library(psych); library(polycor)
source("miFunctions.R")
# Create Output
filename <- "oneACEvc"
sink(paste(filename,".Ro",sep=""), append=FALSE, split=TRUE)
# ----------------------------------------------------------------------------------------------------------------------
# PREPARE DATA
# Load Data
data(twinData)
dim(twinData)
describe(twinData[,1:12], skew=F)
# Select Variables for Analysis
vars <- 'bmi' # list of variables names
nv <- 1 # number of variables
ntv <- nv*2 # number of total variables
selVars <- paste(vars,c(rep(1,nv),rep(2,nv)),sep="")
# Select Data for Analysis
mzData <- subset(twinData, zyg==1, selVars)
dzData <- subset(twinData, zyg==3, selVars)
# Generate Descriptive Statistics
colMeans(mzData,na.rm=TRUE)
colMeans(dzData,na.rm=TRUE)
cov(mzData,use="complete")
cov(dzData,use="complete")
# Set Starting Values
svMe <- 20 # start value for means
svPa <- .2 # start value for path coefficient
svPe <- .5 # start value for path coefficient for e
# ----------------------------------------------------------------------------------------------------------------------
# PREPARE MODEL
# Create Algebra for expected Mean Matrices
meanG <- mxMatrix( type="Full", nrow=1, ncol=ntv, free=TRUE, values=svMe, labels=labVars("mean",vars), name="meanG" )
# Create Matrices for Variance Components
covA <- mxMatrix( type="Symm", nrow=nv, ncol=nv, free=TRUE, values=svPa, label="VA11", name="VA" )
covC <- mxMatrix( type="Symm", nrow=nv, ncol=nv, free=TRUE, values=svPa, label="VC11", name="VC" )
covE <- mxMatrix( type="Symm", nrow=nv, ncol=nv, free=TRUE, values=svPa, label="VE11", name="VE" )
# Create Algebra for expected Variance/Covariance Matrices in MZ & DZ twins
covP <- mxAlgebra( expression= VA+VC+VE, name="V" )
covMZ <- mxAlgebra( expression= VA+VC, name="cMZ" )
covDZ <- mxAlgebra( expression= 0.5%x%VA+ VC, name="cDZ" )
expCovMZ <- mxAlgebra( expression= rbind( cbind(V, cMZ), cbind(t(cMZ), V)), name="expCovMZ" )
expCovDZ <- mxAlgebra( expression= rbind( cbind(V, cDZ), cbind(t(cDZ), V)), name="expCovDZ" )
# Create Data Objects for Multiple Groups
dataMZ <- mxData( observed=mzData, type="raw" )
dataDZ <- mxData( observed=dzData, type="raw" )
# Create Expectation Objects for Multiple Groups
expMZ <- mxExpectationNormal( covariance="expCovMZ", means="meanG", dimnames=selVars )
expDZ <- mxExpectationNormal( covariance="expCovDZ", means="meanG", dimnames=selVars )
funML <- mxFitFunctionML()
# Create Model Objects for Multiple Groups
pars <- list( meanG, covA, covC, covE, covP )
modelMZ <- mxModel( pars, covMZ, expCovMZ, dataMZ, expMZ, funML, name="MZ" )
modelDZ <- mxModel( pars, covDZ, expCovDZ, dataDZ, expDZ, funML, name="DZ" )
multi <- mxFitFunctionMultigroup( c("MZ","DZ") )
# Create Algebra for Unstandardized and Standardized Variance Components
rowUS <- rep('US',nv)
colUS <- rep(c('VA','VC','VE','SA','SC','SE'),each=nv)
estUS <- mxAlgebra( expression=cbind(VA,VC,VE,VA/V,VC/V,VE/V), name="US", dimnames=list(rowUS,colUS) )
# Create Confidence Interval Objects
ciACE <- mxCI( "US[1,1:3]" )
# Build Model with Confidence Intervals
modelACE <- mxModel( "oneACEvc", pars, modelMZ, modelDZ, multi, estUS, ciACE )
# ----------------------------------------------------------------------------------------------------------------------
# RUN MODEL
# Run ACE Model
fitACE <- mxRun( modelACE, intervals=T )
sumACE <- summary( fitACE )
# Compare with Saturated Model
#if saturated model fitted in same session
#mxCompare( fitSAT, fitACE )
#if saturated model prior to genetic model
#lrtSAT(fitACE,4055.9346,1767)
# Print Goodness-of-fit Statistics & Parameter Estimates
fitGofs(fitACE)
fitEstCis(fitACE)
# ----------------------------------------------------------------------------------------------------------------------
# RUN SUBMODELS
# Run AE model
modelAE <- mxModel( fitACE, name="oneAEvc" )
modelAE <- omxSetParameters( modelAE, labels="VC11", free=FALSE, values=0 )
fitAE <- mxRun( modelAE, intervals=T )
fitGofs(fitAE); fitEstCis(fitAE)
# Run CE model
modelCE <- mxModel( fitACE, name="oneCEvc" )
modelCE <- omxSetParameters( modelCE, labels="VA11", free=FALSE, values=0 )
modelCE <- omxSetParameters( modelCE, labels=c("VE11","VC11"), free=TRUE, values=.6 )
fitCE <- mxRun( modelCE, intervals=T )
fitGofs(fitCE); fitEstCis(fitCE)
# Run E model
modelE <- mxModel( fitAE, name="oneEvc" )
modelE <- omxSetParameters( modelE, labels="VA11", free=FALSE, values=0 )
fitE <- mxRun( modelE, intervals=T )
fitGofs(fitE); fitEstCis(fitE)
# Print Comparative Fit Statistics
mxCompare( fitACE, nested <- list(fitAE, fitCE, fitE) )
round(rbind(fitACE$US$result,fitAE$US$result,fitCE$US$result,fitE$US$result),4)
# ----------------------------------------------------------------------------------------------------------------------
sink()
save.image(paste(filename,".Ri",sep=""))