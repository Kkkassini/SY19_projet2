astro <- read.csv('astronomy_train.csv')
#1 GALAXY  2 DAO 3 STAR

v <- var(astro[,-14])

astro <- astro[,-c(1,10)]

class <- astro$class
astro <- astro[,-12]
astro2 <- cbind(astro,class)
levels(class) <- c(1,2,3)
class <- as.factor(class)
astro <- cbind(astro,class)


n <- nrow(astro)
#ntst <- n-round(2*n/3)

set.seed(999)
train <- sample(1:n,round(2*n/3))


astro.train <- astro[train,]
astro.test <- astro[-train,]

astro.scale <- data.frame(scale(astro[,-16]))

astro.scale <- cbind(astro.scale,class)
astro.train.scale <- astro.scale[train,]
astro.test.scale <- astro.scale[-train,]

library(caret) 
K=10 
folds<-createFolds(1:n,K)
ntst <- n/K

# correlation <- cor(as.numeric(astro.scale$class), astro.scale[,-16], use="pairwise.complete.obs") 
# library(corrplot)
# #corrplot(data.frame(correlation))
# correlation<-abs(correlation)
# order(correlation)
# correlation<- correlation[,order(correlation, decreasing=T)]
# correlation
# 
# cor1 <- cor(astro.scale[,-16])
# corrplot(cor1,method="circle")


#plot 
par(mar = c(5.1, 4.1, 4.1, 2.1))

#pca svd

#choose 10 main  0.98675
X <- astro.scale[,-16]
pca3 <- prcomp(X,rank = 15) 

summary(pca3)
screeplot(pca3,type='line',main='ScreePlot',lwd=2, npcs = 15)
X <- pca3$x[,1:10]
rotation <- pca3$rotation[1:10,]

astro.pca <- data.frame(cbind(X, astro$class))
names(astro.pca)[names(astro.pca)=="V11"]="class"

train.pca <- astro.pca[train,]
test.pca <- astro.pca[-train,]
#X.test <- as.matrix(astro.test.scale[,-16])%*%rotation


#Logistic regression
#lineaire
library(nnet)
error_lr <- rep(0,K)
for(i in (1:K)){
  data.train <- astro.pca[-folds[[i]],]
  data.test <- astro.pca[folds[[i]],]
  lr.fit <- multinom( class~.,data=data.train )
  lr.pred <- predict( lr.fit, newdata=data.test[,-11] )
  lr.table <- table( data.test$class, lr.pred )
  error_lr[i] <- ( 1-sum( diag(lr.table) ) / (n/K) )
}
mean(error_lr)
ic.lr <- t.test(error_lr)
ic.lr <- as.numeric(ic.lr$conf.int)


#qda 
#quadratic
library(MASS)
error_qda <- rep(0,K)
for(i in (1:K)){
  data.train <- astro.pca[-folds[[i]],]
  data.test <- astro.pca[folds[[i]],]
  qda.fit <- qda( class~.,data=data.train )
  qda.pred <- predict( qda.fit, newdata=data.test[,-11] )
  qda.table <- table( data.test$class, qda.pred$class )
  error_qda[i] <- (1 - sum(diag(qda.table)) / (n/K))
}
mean(error_qda)
ic.qda <- t.test(error_qda)
ic.qda <- as.numeric(ic.qda$conf.int)



#knn
#non lineaire
library('FNN')
error_knn <- rep(0,K)
for(i in (1:K)){
  data.train <- astro.pca[-folds[[i]],]
  data.test <- astro.pca[folds[[i]],]
  knn.fit <- knn(data.train[,-11], data.test[,-11], data.train[,11], k = 10, prob = FALSE, 
                 algorithm=c("kd_tree", "cover_tree", "brute"))
 # knn.pred <- predict( knn.fit, newdata=data.test[,-11] )
  knn.table <- table( data.test$class, knn.fit )
  error_knn[i] <- (1 - sum(diag(knn.table)) / (n/K))
}
mean(error_knn)
ic.knn <- t.test(error_knn)
ic.knn <- as.numeric(ic.knn$conf.int)



#naive bayes
#assume non correlee
library('e1071')
error_nb <- rep(0,K)
for(i in (1:K)){
  data.train <- astro.pca[-folds[[i]],]
  data.test <- astro.pca[folds[[i]],]
  nb.mod <- naiveBayes(class~.,data = data.train)
  nb.pred <- predict( nb.mod, newdata=data.test[,-11],type = "raw")
  nb.pred <- max.col(nb.pred)
  nb.table <- table( data.test$class, nb.pred )
  error_nb[i] <- (1 - sum(diag(nb.table)) / (n/K))
}
mean(error_nb)
ic.nb <- t.test(error_nb)
ic.nb <- as.numeric(ic.nb$conf.int)


#decision tree
library('rpart')
error_dtree <- rep(0,K)
for(i in (1:K)){
  data.train <- astro2[-folds[[i]],]
  data.test <- astro2[folds[[i]],]
 # data.train[,16] <- as.factor(data.train[,16])
  dtree.mod <- rpart(class ~ ., data = data.train, method="class", parms = list(split = 'gini'),cp = 0.001)
  dtree.pred <- predict( dtree.mod, newdata=data.test[,-16],type = "class")
  dtree.table <- table( data.test$class, dtree.pred )
  error_dtree[i] <- (1 - sum(diag(dtree.table)) / (n/K))
}
mean(error_dtree)
ic.dtree <- t.test(error_dtree)
ic.dtree <- as.numeric(ic.dtree$conf.int)

printcp(dtree.mod)
library(rpart.plot)
rpart.plot(dtree.mod, box.palette="RdBu", shadow.col="gray", fallen.leaves=FALSE)
# plot(tree.fit,margin = 0.05)
# text(tree.fit,pretty=0,cex=0.8)
# plotcp(dtree.mod)

rpart.plot(dtree.mod, box.palette="RdBu", shadow.col="gray", fallen.leaves=FALSE);
plot(dtree.mod,margin = 0.05);
text(dtree.mod,pretty=0,cex=0.8);
plotcp(dtree.mod);


#bagging
library(randomForest)
error_bag <- rep(0,K)
for(i in (1:K)){
  data.train <- astro2[-folds[[i]],]
  data.test <- astro2[folds[[i]],]
  #data.train[,16] <- as.factor(data.train[,16])
  bag.mod <- randomForest(class ~ ., data = data.train, method="class", ntree=500, mtry = 15)
  bag.pred <- predict( bag.mod, newdata=data.test[,-16],type = "class")
  bag.table <- table( data.test$class, bag.pred )
  error_bag[i] <- (1 - sum(diag(bag.table)) / (n/K))
}
mean(error_bag)
ic.bag <- t.test(error_bag)
ic.bag <- as.numeric(ic.bag$conf.int)

#random forest

error_rf <- rep(0,K)
for(i in (1:K)){
  data.train <- astro2[-folds[[i]],]
  data.test <- astro2[folds[[i]],]
  rf.mod <- randomForest(class~., data=data.train)
  rf.pred <- predict( rf.mod, newdata=data.test[,-16],type = "response")
  rf.table <- table( data.test$class, rf.pred )
  error_rf[i] <- (1 - sum(diag(rf.table)) / (n/K))
}
mean(error_rf)
ic.rf <- t.test(error_rf)
ic.rf <- as.numeric(ic.rf$conf.int)





#SVM
library("kernlab")
error_svm <- matrix(0,4,K)
ker <- c('polydot','laplacedot','rbfdot','vanilladot')
for(i in (1:K)){
  data.train <- astro.pca[-folds[[i]],]
  data.test <- astro.pca[folds[[i]],]
  for (j in 1:4) {
      svm.mod <- ksvm(as.factor(class)~ .,data = data.train,
                   type="C-svc",kernel = ker[j] ,C = 100)
      svm.pred <- predict( svm.mod, newdata=data.test[,-16],type = "response")
      svm.table <- table( data.test$class, svm.pred )
      error_svm[j,i] <- (1 - sum(diag(svm.table)) / (n/K))
  }


}
mean(error_svm[1,])
mean(error_svm[2,])
mean(error_svm[3,])
mean(error_svm[4,])
mean_error_svm <- c(mean(error_svm[1,]),mean(error_svm[2,]),mean(error_svm[3,]),mean(error_svm[4,]))


polynomial <- t.test(error_svm[1,])
polynomial <- as.numeric(polynomial$conf.int)
laplacien <- t.test(error_svm[2,])
laplacien <- as.numeric(laplacien$conf.int)
gaussien <- t.test(error_svm[3,])
gaussien <- as.numeric(gaussien$conf.int)
lineaire <- t.test(error_svm[4,])
lineaire <- as.numeric(lineaire$conf.int)
ic.svm <- cbind(polynomial,laplacien,gaussien,lineaire)
error.svm <- rbind(mean_error_svm, ic.svm)
rownames(error.svm) <- c('mean error', 'IC1', 'IC2')

#neural network
library('nnet')
error_nnt <- rep(0,K)
for(i in (1:K)){
  data.train <- astro.pca[-folds[[i]],]
  data.test <- astro.pca[folds[[i]],]
  data.train[,11] <- as.factor(data.train[,11])
  nnt.mod <- nnet(class~., data = data.train, size=5, linout = TRUE)
  nnt.pred <- predict(nnt.mod, newdata=data.test[,-11],type = "class")
  nnt.table <- table( data.test$class, nnt.pred )
  error_nnt[i] <- (1 - sum(diag(nnt.table)) / (n/K))
}
mean(error_nnt)
ic.nnt <- t.test(error_nnt)
ic.nnt <- as.numeric(ic.nnt$conf.int)


# #roc curve
# library(pROC)
# data.train <- astro2[-folds[[i]],]
# data.test <- astro2[folds[[i]],]
# rf.pred_1 <- predict( rf.mod, newdata=data.test[,-16],type = "response")
# roc_curve <- multiclass.roc(data.test$class,as.numeric(rf.pred_1))
# auc(roc_curve)
# rs <- roc_curve[['rocs']]
# plot.roc(rs)
# sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))
# lines.roc(data.test$class,as.numeric(nb.pred),levels = c(1,2,3))

save.image(file = "astro_rapport.RData")

