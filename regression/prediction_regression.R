predict_regression <- function(dataset) {
  # Chargement de l’environnement
  library(e1071)
  load("reg_model.Rdata")
  
  #traitement du dataset
  data <- dataset
  data[, 4] <- as.factor(data[, 4]) # IRR
  data <- dataset[,-c(1,3)] #on retire l'année et le département
  colnames(data)[1] <- "y" #on renomme la variable à prédire
  data[,-c(1,2)] <- scale(data[,-c(1,2)]) #on scale les données quantitatives
  
  
  IRR1 <- as.numeric(data[,"IRR"] == 1)
  IRR2 <- as.numeric(data[,"IRR"] == 2)
  IRR3 <- as.numeric(data[,"IRR"] == 3)
  IRR4 <- as.numeric(data[,"IRR"] == 4)
  IRR5 <- as.numeric(data[,"IRR"] == 5)
  dummyIRR <- cbind(IRR1,IRR2,IRR3,IRR4,IRR5)
  data <- data[,-c(2)]
  data <- cbind(data,dummyIRR) #tableau disjonctif de "IRR"
  
  
  predictions <- predict(reg.fit, newdata=data,type = "response")
  return(predictions)
}