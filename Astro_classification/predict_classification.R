classifieur_astronomie <- function(dataset) {
  # Chargement de l’environnement
  load("class.Rdata")
  library(randomForest)
  predictions <- predict(mode_class, newdata=dataset,type = "class")
  # Mon algorithme qui renvoie les prédictions sur le jeu de données
  # ‘dataset‘ fourni en argument.
  # ...
  return(predictions)
}
