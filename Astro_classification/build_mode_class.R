astro <- read.csv('astronomy_train.csv')
#1 GALAXY  2 DAO 3 STAR

#astro <- astro[,-c(1,10)]
n <- nrow(astro)
set.seed(999)
class <- astro$class
astro <- astro[,-16]
astro <- cbind(astro,class)
library(caret) 
K=10 
folds<-createFolds(1:n,K)
# levels(class) <- c(1,2,3)
# class <- as.factor(class)
# astro <- cbind(astro,class)
library(randomForest)
error_rf <- rep(0,K)
rf.mod <- list()
for(i in (1:K)){
  data.train <- astro[-folds[[i]],]
  data.test <- astro[folds[[i]],]
  #data.train[,16] <- as.factor(data.train[,16])
  rf.mod[[i]] <- randomForest(class ~ ., data = data.train, method="class", ntree=500)
  rf.pred <- predict( rf.mod[[i]], newdata=data.test[,-18],type = "class")
  rf.table <- table( data.test$class, rf.pred )
  error_rf[i] <- (1 - sum(diag(rf.table)) / (n/K))
}
mean(error_rf)
ic.rf <- t.test(error_rf)
ic.rf <- as.numeric(ic.rf$conf.int)
best_ind <- which.min(error_rf)
mode_class <- rf.mod[[best_ind]]
save(mode_class, file = "class.RData")



