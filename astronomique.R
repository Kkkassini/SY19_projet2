data <- read.csv('astronomy_train.csv')
#1 GALAXY  2 DAO 3 STAR

v <- var(data[,-14])

data <- data[,-c(1,10)]

class <- data$class
data <- data[,-12]

levels(class) <- c(1,2,3)
class <- as.factor(class)
data <- cbind(data,class)
n <- nrow(data)
ntst <- n-round(2*n/3)
train <- sample(1:n,round(2*n/3))


data.train <- data[train,]
data.test <- data[-train,]

data.scale <- data.frame(scale(data[,-16]))
mode(data.scale)
data.scale <- cbind(data.scale,class)
data.train.scale <- data.scale[train,]
data.test.scale <- data.scale[-train,]

library(caret) 
K=10 
folds<-createFolds(1:n,K)

which(sapply(data.scale, is.factor))

correlation <- cor(as.numeric(data.scale$class), data.scale[,-16], use="pairwise.complete.obs") 
library(corrplot)
#corrplot(data.frame(correlation))
correlation<-abs(correlation)
order(correlation)
correlation<- correlation[,order(correlation, decreasing=T)]
correlation

cor1 <- cor(data.scale[,-16])
corrplot(cor1,method="circle")

# u,g,r,i,z
#plate,mjd,specobjid

library('psych')
cor2 <- corr.test(data[,-16], use = "complete", method = "pearson", adjust = "none")
cor.plot(cor1)


#int run rerun camcol field plate mjd fiberid
#real objid ra dec u g r i z specobjid redshift 
int <- c(9,10,11,12,16,17,18)
real <- c(1,2,3,4,5,6,7,8,13,15)
#iscm qualitative
#acp réel  80
#acm entier
#Pfamd analyse profonde
#analyse temporaire
#factor mineur library


library(FactoMineR)
pca <- PCA(data.scale[, -16],ncp = 15)
plot(pca$eig[,3], type='l', ylab='cumulative percentage of variance',
     xlab="components")
#res.mca = MCA(data, abbrev=TRUE)
U <- as.matrix(pca$svd$U)
V <- as.matrix(pca$svd$V)
library('factoextra')
get_eigenvalue(pca)
get_pca_var(pca)
dim(t(V))
a <-U%*%t(V)-data.scale[,-16]
plot(pca)
screeplot(pca,type='line',main='ScreePlot',lwd=2)
pca$svd$V*data.scale[,-16]-pca$svd$U

X <- data.scale[,-16]
pca2<-princomp(X)
as.matrix(X)%*%pca2$loadings-pca2$scores
X <- X*pca2$loadings
screeplot(pca2,type='line',main='ScreePlot',lwd=2)


#pca svd
X <- data.train.scale[,-16]
pca3 <- prcomp(X,rank. = 15)  
screeplot(pca3,type='line',main='ScreePlot',lwd=2, npcs = 15)
#biplot(pca3)
#fviz_pca_var(pca3, col.var = "black")

pca3 <- prcomp(X,rank. = 10) 
X <- pca3$x
train.pca <- data.frame(cbind(X,data.train.scale$class))
names(train.pca)[names(train.pca)=="V11"]="class"
X.test <- as.matrix(data.test.scale[,-16])%*%pca3$rotation
test.pca <- data.frame(cbind(X.test,data.test.scale$class))
names(test.pca)[names(test.pca)=="V11"]="class"


#Logistic regression
library(nnet)
lr.fit <- multinom(class~.,data=data.train.scale)
lr.pred <- predict( lr.fit, newdata= data.test.scale[,-16])
perf.lr <-table(data.test.scale$class,lr.pred)
1-sum(diag(perf))/ntst
perf.lr

#qda 
library(MASS)

data.qda<- qda(class~.,data=data.train.scale)
pred.qda<-predict(data.qda,newdata=data.test.scale)
perf.qda <-table(data.test.scale$class,pred.qda$class)
1-sum(diag(perf.qda))/ntst
perf.qda

data.qda2 <- qda(class~.,data=data.frame(train.pca))
pred.qda2 <- predict(data.qda2,newdata=test.pca)
perf.qda2 <- table(test.pca$class,pred.qda2$class)
1-sum(diag(perf.qda2))/ntst
perf.qda2




#knn
library('FNN')
knn <- knn(train.pca[,-11], test.pca[,-11], train.pca[,11], k = 1, prob = FALSE, algorithm=c("kd_tree", 
                                                      "cover_tree", "brute"))
length(which(knn==test.pca[,11]))/length(test.pca[,11])
perf.knn <- table(test.pca$class,knn)
1-sum(diag(perf.knn))/ntst
perf.knn

#naive bayes
#assume non correlee
library('e1071')

nbayes <- naiveBayes(class~.,data = train.pca)
nbayes <- predict(nbayes, newdata = test.pca[,-11],type = "raw")
nbayes <- max.col(nbayes)
perf.bayes <- table(test.pca$class,nbayes)
1-sum(diag(perf.bayes))/ntst
perf.bayes


#decision tree
library('rpart')
tree.fit <- rpart(class ~ ., data = data.train, method="class", parms = list(split = 'gini'),cp = 0)
plot(tree.fit,margin = 0.05)
text(tree.fit,pretty=0,cex=0.8)
plotcp(fit)

library(rpart.plot)
rpart.plot(tree.fit, box.palette="RdBu", shadow.col="gray", fallen.leaves=FALSE)

yhat=predict(tree.fit,newdata=data.test[,-16],type='class')
table(data.test[,16],yhat)
err<-1-mean(data.test[,16]==yhat)
err


#random forest
library(randomForest)
fit.rf <- randomForest(class~., data=data.train)
yhat.rf <- predict(fit.rf, newdata=data.test, type="response")
CM.rf <- table(data.test$class, yhat.rf)
err.rf <- 1-mean(data.test$class==yhat.rf)
err.rf




#SVM
library("kernlab")

svmfit<-ksvm(as.factor(class)~ .,data=data.train,
             type="C-svc",kernel="polydot",C=10)

yhat.svm <- predict(svmfit,newdata = data.test[,-16])

CM.svm <- table(data.test$class, yhat.svm)
err.svm <- 1-mean(data.test$class==yhat.svm)
err.svm
CM.svm


#neural network
library('nnet')
nn1<- nnet(class~., data = data.train.scale, size=5, linout = TRUE)
pred1<- predict(nn1,newdata=data.test.scale[,-16],type = "class")
CM.nn <- table(data.test.scale$class, pred1)
err.nn <- 1-mean(data.test$class==pred1)
err.nn
CM.nn
confusionMatrix(CM.nn)
summary(nn1)
print(nn1)



data.train$class <- as.numeric(data.train$class)
data.test$class <- as.numeric(data.test$class)

library('keras')
model <- keras_model_sequential()
model %>% layer_dense(units = 15, activation = 'relu', input_shape = 15) %>%
  layer_dense(units = 20, activation = 'relu',name="cache1") %>%
  layer_dense(units = 5, activation = 'relu',name="cache2") %>%
  layer_dense(units = 1, activation = 'linear',name="sortie")
model %>% compile(loss = 'mean_squared_error', optimizer = optimizer_rmsprop())
history <- model %>% fit(as.matrix(data.train[,-16]), as.matrix(data.train$class),
                         epochs = 2000, batch_size = 30)
x=seq(-2,3,0.01)
pred <- predict(model, as.matrix(data.test[,-16]))

summary(model)
pred

model %>% predict_classes( as.matrix(data.test[,-16]))

rm(model)

