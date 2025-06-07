library(torch)

# 1. Create synthetic binary classification data
set.seed(123)
timesteps <- 10000
data <- sin(1:timesteps * 0.2) + rnorm(timesteps, sd = 0.1)

# Create binary labels: 1 if next value > 0, else 0
create_sequences <- function(data, lookback = 20) {
  x <- list()
  y <- numeric()
  for (i in 1:(length(data) - lookback)) {
    x[[i]] <- data[i:(i + lookback - 1)]
    y[i] <- ifelse(data[i + lookback] > 0, 1, 0)
  }
  x_tensor <- torch_tensor(do.call(rbind, x), dtype = torch_float())$unsqueeze(3)
  y_tensor <- torch_tensor(y, dtype = torch_float())
  list(x = x_tensor, y = y_tensor)
}

seq_data <- create_sequences(data, lookback = 20)
x <- seq_data$x
y <- seq_data$y

# ✅ Automatically determine train/test split
total_samples <- x$size()[1]
train_frac <- 0.8
train_size <- floor(train_frac * total_samples)

x_train <- x[1:train_size,,]
y_train <- y[1:train_size]
x_test <- x[(train_size + 1):total_samples,,]
y_test <- y[(train_size + 1):total_samples]

# 2. Define binary classification LSTM model
LSTMClassifier <- nn_module(
  "LSTMClassifier",
  initialize = function(input_size, hidden_size) {
    self$lstm <- nn_lstm(input_size = input_size, hidden_size = hidden_size, batch_first = TRUE)
    self$fc <- nn_linear(hidden_size, 1)
    self$sigmoid <- nn_sigmoid()
  },
  forward = function(x) {
    lstm_out <- self$lstm(x)[[1]]
    last_hidden <- lstm_out[, -1, ]
    out <- self$fc(last_hidden)
    self$sigmoid(out)
  }
)

# 3. Instantiate model
model <- LSTMClassifier$new(input_size = 1, hidden_size = 64)
optimizer <- optim_adam(model$parameters, lr = 0.01)
loss_fn <- nn_bce_loss()

# 4. Train
for (epoch in 1:10) {
  model$train()
  optimizer$zero_grad()
  output <- model$forward(x_train)
  loss <- loss_fn(output$squeeze(), y_train)
  loss$backward()
  optimizer$step()
  cat(sprintf("Epoch %d - Loss: %.4f\n", epoch, loss$item()))
}

# 5. Predict
model$eval()
with_no_grad({
  prob <- model$forward(x_test)$squeeze()
  preds <- as.numeric(prob > 0.5)
  actual <- as.numeric(y_test)
})

# 6. Accuracy
accuracy <- mean(preds == actual)
cat(sprintf("Test Accuracy: %.2f%%\n", accuracy * 100))

# 7. Plot
plot(actual, type = 'l', col = 'blue', ylab = 'Class', main = 'LSTM Classification')
lines(preds, col = 'red')
legend("bottomleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1)
