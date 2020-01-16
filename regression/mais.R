setwd("D:/UTC/GI05/SY19/sy19_projet2")
library(randomForest)
library(splines)
library(rpart)
library(MASS)
library(gam)
library(glmnet)
library(e1071)

data <- read.csv(file = "./mais_train.csv", row.names = 1)
lapply(data,class)
colnames(data)[2] <- "y"
data[, 1] <- as.factor(data[, 1]) # year_harvest
data[, 3] <- as.factor(data[, 3]) # NUMD
data[, 4] <- as.factor(data[, 4]) # IRR
scaledData <- data
scaledData[,-c(1,2,3,4)] <- scale(data[,-c(1,2,3,4)])
p <- ncol(data) - 1
n <- nrow(data)

IRR1 <- as.numeric(data[,"IRR"] == 1)
IRR2 <- as.numeric(data[,"IRR"] == 2)
IRR3 <- as.numeric(data[,"IRR"] == 3)
IRR4 <- as.numeric(data[,"IRR"] == 4)
IRR5 <- as.numeric(data[,"IRR"] == 5)
dummyIRR <- cbind(IRR1,IRR2,IRR3,IRR4,IRR5)

dataDummyIrr <- cbind(data[,-c(4)],dummyIRR)
scaledDataDummyIrr <- dataDummyIrr
scaledDataDummyIrr[,-c(1,2,3,58,59,60,61,62)] <- scale(dataDummyIrr[,-c(1,2,3,58,59,60,61,62)])

scaledDataDummyIrrWithoutYearAndDep <- scaledDataDummyIrr[,-c(1,3)]

remove(IRR1,IRR2,IRR3,IRR4,IRR5,dummyIRR)

pca <- princomp(scaledData[,-c(1,2,3,4)])
#59 variables : 58-year-numd-y-Irr+4*dummyIrr
summary(pca)
(pca$scores)
acpdata <- cbind(data[,c(1,2,3,4)],pca$scores)
for(i in 5:ncol(acpdata)) {
  colnames(acpdata)[i] <- paste("Z",i-4,sep="")
}
acpdata <- as.data.frame.matrix(acpdata)

#returns formula with n first pcomps
get_pcomps <- function(n) {
  formula = "y~year_harvest+NUMD+IRR+Z1"
  for(i in 2:n) {
    formula = paste(formula,"+Z",i,sep="")
  }
  return(as.formula(formula))
}


remove(pca,i)

Kfolds <- 10
folds <- sample(1:Kfolds, n, replace=TRUE)

mse.regLin <- rep(0,Kfolds)
mse.tree <- rep(0,Kfolds)
mse.randomForest <- rep(0,Kfolds)
mse.bagging <- rep(0,Kfolds)
mse.glm <- rep(0,Kfolds)
mse.glmpca <- rep(0,Kfolds)
mse.lasso <- rep(0,Kfolds)
mse.radial.svr <- rep(0,Kfolds)

for(k in 1 : Kfolds) {
  #non-parametric models : no intern cross-val
  
  #I.Linear Regression
  fit.regLin <- lm(y~ ., data = scaledDataDummyIrrWithoutYearAndDep, subset=(folds!=k))
  pred.regLin <- predict(fit.regLin, newdata=scaledDataDummyIrrWithoutYearAndDep[folds==k,-c(1)])
  mse.regLin[k] <- (mean((pred.regLin - scaledDataDummyIrrWithoutYearAndDep[folds==k,"y"])^2))
  
  #II.Decision Tree
  #get optimal cp value by intern cross-val :
  fit.tree <- rpart(y~year_harvest+NUMD, data=data, subset = (folds!=k),  method="anova", control=rpart.control(xval = Kfolds, cp=0.00))
  plotcp(fit.tree)
  i.min<-which.min(fit.tree$cptable[,4])
  tree.cp.opt<-fit.tree$cptable[i.min,1]
  pruned_tree <- prune(fit.tree, cp=tree.cp.opt)
  pred.tree <- predict(pruned_tree, newdata=data[folds==k,-c(2)])
  mse.tree[k] <- (mean((pred.tree - data[folds==k,"y"])^2))
  
  #III.Random Forest
  fit.randomForest <- randomForest(y~., data=scaledData[folds!=k,-c(1,3)])
  pred.randomForest <- predict(fit.randomForest,newdata=scaledData[folds==k,-c(2)],type='response')
  mse.randomForest[k] <- (mean((pred.randomForest - scaledData[folds==k,"y"])^2))
  
  #IV.Bagging
  fit.bagging <- randomForest(y~., data=scaledData[folds!=k,-c(1,3)],mtry=p)
  pred.bagging <- predict(fit.bagging, newdata=scaledData[folds==k,-c(2)],type='response')
  mse.bagging[k] <- (mean((pred.bagging - scaledData[folds==k,"y"])^2))
  
  #V.glm with both factors and reals :
  fit.glm <- glm(y~year_harvest+NUMD+IRR+ETP_1+ETP_2+ETP_3+ETP_4+ETP_5+ETP_6+ETP_7+ETP_8+ETP_9+PR_1+PR_2+PR_3
                 +PR_4+PR_5+PR_6+PR_7+PR_8+PR_9+RV_1+RV_2+RV_3+RV_4+RV_5+RV_6+RV_7+RV_8+RV_9+SeqPR_1+SeqPR_2
                 +SeqPR_3+SeqPR_4+SeqPR_5+SeqPR_6+SeqPR_7+SeqPR_8+SeqPR_9+Tn_1+Tn_2+Tn_3+Tn_4+Tn_5+Tn_6+Tn_7+Tn_8
                 +Tn_9+Tx_1+Tx_2+Tx_3+Tx_4+Tx_5+Tx_6+Tx_7+Tx_8+Tx_9, data=scaledData,subset=(folds!=k))
  pred.glm <- predict(fit.glm, newdata=scaledData[folds==k,-c(2)])
  mse.glm[k] <- (mean((pred.glm - scaledData[folds==k,"y"])^2))
  
  #glm after pca
  #cross val to get optimal number of pcomp :
  interndata = acpdata[folds!=k,]
  intern_n <- nrow(interndata)
  intern.Kfolds <- 10
  intern.folds = sample(1:intern.Kfolds, intern_n, replace = TRUE)
  CV.glmpca <- rep(0,54)
  for(intern.k in 1:intern.Kfolds) {
    print(intern.k)
    for(i in 1:54) {
      fit.glmpca <- glm(get_pcomps(i), data=interndata,subset=(intern.folds!=intern.Kfolds))
      pred.glmpca <- predict(fit.glmpca, newdata=interndata[intern.folds==intern.Kfolds,-c(2)])
      CV.glmpca[i] <- CV.glmpca[i] + (mean((pred.glmpca - interndata[intern.folds==intern.Kfolds,"y"])^2))
    }
  }
  CV.glmpca <- CV.glmpca/intern.Kfolds
  npcomp <- which.min(CV.glmpca)
  fit.glmpca <- glm(get_pcomps(npcomp), data=acpdata,subset=(folds!=k))
  pred.glmpca <- predict(fit.glmpca, newdata=acpdata[folds==k,-c(2)])
  mse.glmpca[k] <- (mean((pred.glmpca - acpdata[folds==k,"y"])^2))
  
  #VI.Lasso regression :
  cv.lasso<-cv.glmnet(as.matrix(scaledData[folds!=k,-c(1,2,3,4)]),as.matrix(scaledData$y[folds!=k]),alpha=1)
  fit.lasso <- glmnet(as.matrix(scaledData[folds!=k,-c(1,2,3,4)]),as.matrix(scaledData$y[folds!=k]),lambda=cv.lasso$lambda.min,alpha=1)
  pred.lasso <- predict(fit.lasso,s=cv.lasso$lambda.min,newx=as.matrix(scaledData[folds==k,-c(1,2,3,4)]))
  mse.lasso[k] <- (mean((pred.lasso - scaledData[folds==k,"y"])^2))
  
  #VII.SVR
  #radial basis
  #optimal parameters : epsilon = 0.2, cost=2
  #optimal.radial.svr = tune(svm, y~., data=scaledDataDummyIrrWithoutYearAndDep[folds!=k,],
                            #ranges=list(epsilon=seq(0,2,0.2), cost=1:10))
  fit.radial.svr <- svm(y~., data=scaledDataDummyIrrWithoutYearAndDep[folds!=k,],
                        epsilon=optimal.radial.svr$best.parameters[,1],cost=optimal.radial.svr$best.parameters[,2])
  pred.radial.svr = predict(fit.radial.svr, newdata=scaledDataDummyIrrWithoutYearAndDep[folds==k,-c(1)])
  mse.radial.svr[k] <- mean((pred.radial.svr - scaledDataDummyIrrWithoutYearAndDep$y[folds==k])^2)
}


#au lieu du boxplot : plot les IC
boxplot(mse.regLin,mse.tree,mse.randomForest,mse.bagging,mse.glm,mse.glmpca,mse.lasso,mse.radial.svr,optimalSmooth[2,])


#we choose SVR

#need to tune hyperparameters for SVR.
optimal.radial.svr = tune(svm, y~., data=scaledDataDummyIrrWithoutYearAndDep,
                          ranges=list(epsilon=seq(0,2,0.2), cost=1:10))


#pas pu tester tous lesmodèles possibles : pas testé ridge, pas testé elasticnet, natural et smooth splines,

#Pas de splines multidimensionnels : dimension de l'espace des prédicteurs tro pélevée et
#taille du dataset trop petite



optimalSmooth = matrix(0,2,19)
optimalSmooth[1,]=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
optimalSmooth[2,]=c(0.9735438,0.9333093,0.8714166,0.8640651,0.854973,0.8370029,0.8101192,0.7790544,0.7463693,0.7414166,0.73439577,0.7345332,0.735001109,0.7368100,0.7350548,0.7329999,0.7311002,0.7225448,0.7050164)





#essayer la regression lineaire mixed : lme4. Supposition : les variables qualitatives peuvent se regrouper dans des classes inconnues