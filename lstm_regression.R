    # Install and load torch
    library(torch)
    
    # 1. Create synthetic time series data
    # Generate synthetic time series data
    set.seed(123)
    timesteps <- 1000
    data <- sin(1:timesteps * 0.02) + rnorm(timesteps, sd = 0.1)
    
    # Create input sequences
    create_sequences <- function(data, lookback = 20) {
      x <- list()
      y <- numeric()
      for (i in 1:(length(data) - lookback)) {
        x[[i]] <- data[i:(i + lookback - 1)]
        y[i] <- data[i + lookback]
      }
      x_tensor <- torch_tensor(do.call(rbind, x), dtype = torch_float())$unsqueeze(3)
      y_tensor <- torch_tensor(y, dtype = torch_float())
      list(x = x_tensor, y = y_tensor)
    }
    
    seq_data <- create_sequences(data, lookback = 20)
    x <- seq_data$x
    y <- seq_data$y
    
    x_train <- x[1:800,,]
    y_train <- y[1:800]
    x_test <- x[801:980,,]
    y_test <- y[801:980]
    
    # Define LSTM model class
    LSTMModel <- nn_module(
      "LSTMModel",
      initialize = function(input_size, hidden_size, output_size) {
        self$lstm <- nn_lstm(input_size = input_size, hidden_size = hidden_size, batch_first = TRUE)
        self$fc <- nn_linear(hidden_size, output_size)
      },
      forward = function(x) {
        out <- self$lstm(x)
        h_n <- out[[1]][, -1, ]
        self$fc(h_n)
      }
    )
    
    # ✅ Instantiate model using $new() — CRITICAL
    model <- LSTMModel$new(input_size = 1, hidden_size = 64, output_size = 1)
    optimizer <- optim_adam(model$parameters, lr = 0.01)
    loss_fn <- nn_mse_loss()
    exists("model")
    
    # Train
    for (epoch in 1:10) {
      model$train()
      optimizer$zero_grad()
      output <- model$forward(x_train)
      loss <- loss_fn(output$squeeze(), y_train)
      loss$backward()
      optimizer$step()
      cat(sprintf("Epoch %d - Loss: %.4f\n", epoch, loss$item()))
    }
    
    # Predict
    model$eval()
    with_no_grad({
      predictions <- model$forward(x_test)#$squeeze()$cpu()$numpy()
    })
    
    # Plot
    plot(as.numeric(y_test), type = 'l', col = 'blue', ylab = 'Value', main = 'LSTM Forecast')
    lines(predictions, col = 'red')
    legend("bottomleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1)
