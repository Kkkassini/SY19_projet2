library(keras)
library(formattable)
library(tidyverse)
# list of objects to modle
object_list <- c("car", "cat", "flower")
library(caret)
library(EBImage)
library(knitr)
# image size to scale down to (original images are 100 x 100 px)
img_width <- 70
img_height <- 70
target_size <- c(img_width, img_height)



# define batch size and number of epochs
batch_size <- 32
epochs <- 100


# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "C:/Users/xingjian/Desktop/sy19_projet/images/images_train"
valid_image_files_path <- "C:/Users/xingjian/Desktop/sy19_projet/images/images_validation"
test_image_files_path <- "C:/Users/xingjian/Desktop/sy19_projet/images/images_test"

train_data_gen = image_data_generator(
  rescale = 1/255, #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
  
  featurewise_center = TRUE,
  featurewise_std_normalization = TRUE,
  rotation_range = 30,
  width_shift_range = 0.20,
  height_shift_range = 0.20,
  horizontal_flip = TRUE
)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)  


test_data_gen <- image_data_generator(
  rescale = 1/255
)


# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = object_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = object_list,
                                                    seed = 42)

test_image_array_gen <- flow_images_from_directory(test_image_files_path, 
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = NULL,
                                                   color_mode = "rgb",
                                                   batch_size = batch_size,
                                                   seed = 42,
                                                   shuffle = FALSE)

table(factor(train_image_array_gen$classes))
table(factor(valid_image_array_gen$classes))
table(factor(test_image_array_gen$classes))


# number of training samples
train_samples <- train_image_array_gen$n

# number of validation samples
valid_samples <- valid_image_array_gen$n


test_samples <- test_image_array_gen$n





#Defining the Model

model<-keras_model_sequential()



#Configuring the Model
model %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3),padding="same",
                input_shape=c(70,70,3)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter=48 , kernel_size=c(3,3),padding="same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3) ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  #flatten the input
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  #output layer-3 classes-3 units
  layer_dense(3) %>%
  
  #applying softmax nonlinear activation function to the output layer to calculate
  #cross-entropy
  layer_activation("softmax") #for computing Probabilities of classes-"logit(log probabilities)


#Optimizer -rmsProp to do parameter updates 
opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)


#Compiling the Model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

#Summary of the Model and its Architecture
summary(model)


#-----------------fit the model-------------------------------
hist <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / train_image_array_gen$batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / valid_image_array_gen$batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("C:/Users/xingjian/Desktop/sy19_projet/fruits_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "C:/Users/xingjian/Desktop/sy19_projet/logs")
  )
)

plot(hist)

#------------------predictions---------------------------

predictions1 <- model %>% predict_generator(test_image_array_gen, steps=test_image_array_gen$n/batch_size+1, verbose=1)

colnames(predictions1) <- c("car","cat","flower")

pred_labels<-colnames(predictions1)[apply(predictions1,1,which.max)]
proba<-apply(predictions1, 1, max)
stat_df <- as.data.frame(cbind(test_image_array_gen$filenames, round(proba*100,2), pred_labels))
colnames(stat_df) <- c("filename","proba","class")
stat_df

fname<-as.factor(stat_df$filename)
y_label = c()
for(i in 1:length(fname)){
  if(grepl("cat", fname[i])==TRUE){
    y_label[i] <- "cat"
  }else if (grepl("flower", fname[i])==TRUE){
    y_label[i] <- "flower"
  }else{
    y_label[i] <- "car"
  }
}

y_label <- as.factor(y_label)

y_pred <- as.factor(stat_df$class)

cm <- confusionMatrix(data = y_pred, reference = y_label)
cm$table
#------------------data augmentation----------------------
datagen <- image_data_generator(
  featurewise_center = TRUE,
  featurewise_std_normalization = TRUE,
  rotation_range = 20,
  width_shift_range = 0.30,
  height_shift_range = 0.30,
  horizontal_flip = TRUE
)
datagen %>% fit_image_data_generator(train_image_array_gen)
#------------------save/load model------------------------
save_model_hdf5(model,filepath = "C:/Users/xingjian/Desktop/sy19_projet/model_cnn70", overwrite = TRUE,
                include_optimizer = TRUE)

model <- load_model_hdf5(filepath = "C:/Users/xingjian/Desktop/sy19_projet/model_cnn",compile = T)
