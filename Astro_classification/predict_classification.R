classifieur_astronomie <- function(dataset) {
  # Chargement de l’environnement
  load("class.Rdata")
  library(randomForest)
  predictions <- predict(mode_class, newdata=dataset,type = "class")
  return(predictions)
}
