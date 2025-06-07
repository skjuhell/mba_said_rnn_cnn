library(torch)
library(torchvision)
library(ggplot2)
library(magrittr)
library(cowplot)

# Define CNN model
net <- nn_module(
  "SimpleCNN",
  initialize = function() {
    self$conv1 <- nn_conv2d(3, 16, kernel_size = 3, padding = 1)
    self$pool <- nn_max_pool2d(kernel_size = 2)
    self$conv2 <- nn_conv2d(16, 32, kernel_size = 3, padding = 1)
    self$fc1 <- nn_linear(32 * 8 * 8, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x %>%
      self$conv1() %>%
      nnf_relu() %>%
      self$pool() %>%
      self$conv2() %>%
      nnf_relu() %>%
      self$pool() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2()
  }
)

# Load CIFAR-10 dataset
train_ds <- cifar10_dataset(
  root = tempdir(),
  train = TRUE,
  download = TRUE,
  transform = transform_to_tensor
)
test_ds <- cifar10_dataset(
  root = tempdir(),
  train = FALSE,
  download = TRUE,
  transform = transform_to_tensor
)

# Data loaders
train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 64)

# Set up model, loss, optimizer
model <- net()
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
if (cuda_is_available()) {
  device <- torch_device("cuda")
} else {
  device <- torch_device("cpu")
}

optimizer <- optim_adam(model$parameters, lr = 0.001)
criterion <- nn_cross_entropy_loss()

# Train for 1 epoch
model$train()
train_iter <- dataloader_make_iter(train_dl)

while (TRUE) {
  batch <- tryCatch(
    dataloader_next(train_iter),
    error = function(e) NULL
  )
  
  if (is.null(batch)) break  # end of dataset
  
  inputs <- batch[[1]]$to(dtype = torch_float(), device = device)
  labels <- batch[[2]]$to(dtype = torch_long(), device = device)
  
  optimizer$zero_grad()
  outputs <- model(inputs)
  loss <- criterion(outputs, labels)
  loss$backward()
  optimizer$step()
}

cat("Training complete\n")

# CIFAR-10 class names
classes <- c("airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck")

# Pick 5 images from test set and classify
model$eval()
batch <- dataloader_make_iter(test_dl) %>% dataloader_next()
imgs <- batch[[1]][1:5,,,]$to(device = device)
labels <- batch[[2]][1:5]$to(device = device)
preds <- model(imgs)$argmax(dim = 2)

# Move to CPU and convert to R
imgs_cpu <- imgs$to(device = "cpu")
labels <- as.integer(labels)
preds <- as.integer(preds)

# Create ggplot tiles


plots <- lapply(1:5, function(i) {
  img_tensor <- imgs_cpu[i,,,]$permute(c(2, 3, 1))  # HWC (32,32,3)
  img_arr <- as.array(img_tensor)
  
  df <- as.data.frame(as.table(img_arr[,,1]))
  df$G <- as.vector(img_arr[,,2])
  df$B <- as.vector(img_arr[,,3])
  colnames(df) <- c("y", "x", "R", "G", "B")
  df$x <- as.integer(df$x)
  df$y <- as.integer(df$y)
  df$hex <- rgb(df$R, df$G, df$B)
  
  true_label <- classes[labels[i]]
  pred_label <- classes[preds[i]]
  
  ggplot(df, aes(x = x, y = y, fill = hex)) +
    geom_tile() +
    scale_fill_identity() +
    scale_y_reverse() +
    coord_equal() +
    theme_void() +
    ggtitle(paste0("True: ", true_label, "\nPred: ", pred_label)) +
    theme(plot.title = element_text(size = 10, hjust = 0.5))
})

# Plot 5 images side-by-side
library(cowplot)
plot_grid(plotlist = plots, ncol = 5)

