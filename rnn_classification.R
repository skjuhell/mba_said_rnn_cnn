# Install required libraries if not already installed
required_packages <- c("keras", "tidyverse", "caret")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# Load the dataset
weather_data <- read_csv("../input//seattleWeather_1948-2017.csv")

# View data
head(weather_data)

# Set model parameters
max_len <- 6
batch_size <- 32
total_epochs <- 15
set.seed(123)

# Select target variable
rain <- weather_data$RAIN

# Summarize target
table(rain)

# Generate overlapping sequences
start_indexes <- seq(1, length(rain) - (max_len + 1), by = 3)
weather_matrix <- matrix(nrow = length(start_indexes), ncol = max_len + 1)
for (i in 1:length(start_indexes)){
  weather_matrix[i,] <- rain[start_indexes[i]:(start_indexes[i] + max_len)]
}

# Ensure numeric format
weather_matrix <- weather_matrix * 1
if(anyNA(weather_matrix)){
  weather_matrix <- na.omit(weather_matrix)
}

# Split input (X) and output (y)
X <- weather_matrix[,-ncol(weather_matrix)]
y <- weather_matrix[,ncol(weather_matrix)]

# Create training/test split
training_index <- createDataPartition(y, p = .9, list = FALSE, times = 1)
X_train <- array(X[training_index,], dim = c(length(training_index), max_len, 1))
y_train <- y[training_index]
X_test <- array(X[-training_index,], dim = c(length(y) - length(training_index), max_len, 1))
y_test <- y[-training_index]

# Build model
model <- keras_model_sequential()

# Train model
trained_model <- model %>% fit(
  x = X_train,
  y = y_train,
  batch_size = batch_size,
  epochs = total_epochs,
  validation_split = 0.1
)

# Evaluate performance
classes <- model %>% predict_classes(X_test, batch_size = batch_size)
table(y_test, classes)

# Baseline: predict "same as yesterday"
day_before <- X_test[,max_len - 1,1]
table(y_test, day_before)

# Accuracy
sum(day_before == classes) / length(classes)
