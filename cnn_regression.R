library(torch)
library(magick)
library(jsonlite)
library(ggplot2)

# Set longer timeout and create working dirs
options(timeout = max(600, getOption("timeout")))
dir.create("coco_data", showWarnings = FALSE)
setwd("coco_data")
dir.create("images", showWarnings = FALSE)

# Download using curl (macOS-friendly)
if (!file.exists("val2017.zip")) {
  cat("Downloading COCO validation images...\n")
  system("curl -L -C - -o val2017.zip http://images.cocodataset.org/zips/val2017.zip")
  # For Windows (uncomment the line below if curl doesn't work):
  # download.file("http://images.cocodataset.org/zips/val2017.zip", destfile = "val2017.zip", mode = "wb")
}

if (!file.exists("annotations_trainval2017.zip")) {
  cat("Downloading COCO annotations...\n")
  system("curl -L -C - -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
  # For Windows (uncomment the line below if curl doesn't work):
  # download.file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", destfile = "annotations_trainval2017.zip", mode = "wb")
}

# Unzip datasets
if (!dir.exists("val2017")) unzip("val2017.zip")
if (!dir.exists("annotations")) unzip("annotations_trainval2017.zip")

# Load annotations
anno <- fromJSON("annotations/instances_val2017.json")
cat_id <- anno$categories$id[anno$categories$name == "dog"]

cat_anno <- anno$annotations[anno$annotations$category_id == cat_id, ]

imgs <- anno$images
img_info <- imgs[imgs$id %in% cat_anno$image_id, ]

img_ids <- img_info$id

# Prepare image & label tensors
img_size <- 64
images <- torch_zeros(length(img_ids), 3, img_size, img_size)
labels <- torch_zeros(length(img_ids), 4)

for (i in seq_along(img_ids)) {
  img_id <- img_ids[i]
  info <- img_info[i, ]  # because it's a data.frame row
  file <- sprintf("val2017/%012d.jpg", img_id)
  if (!file.exists(file)) next
  
  img <- image_read(file)
  img <- image_resize(img, paste0(img_size, "x", img_size, "!"))
  arr <- as.numeric(image_data(img)) / 255
  arr <- array(arr, dim = c(3, img_size, img_size))
  images[i,,,] <- torch_tensor(arr)
  
  bbox <- subset(cat_anno, image_id == img_id)$bbox[[1]]
  x <- bbox[1] / info$width
  y <- bbox[2] / info$height
  w <- bbox[3] / info$width
  h <- bbox[4] / info$height
  labels[i,] <- torch_tensor(c(x, y, w, h))
}


# Define dataset and dataloader
dataset <- dataset(
  initialize = function(x, y) {
    self$x <- x
    self$y <- y
  },
  .getitem = function(i) list(x = self$x[i,..], y = self$y[i,]),
  .length = function() self$x$size()[[1]]
)
set.seed(42)  # for reproducibility

n <- images$size(1)
train_frac <- 0.8
train_idx <- sample(1:n, size = floor(train_frac * n))
test_idx <- setdiff(1:n, train_idx)

train_ds <- dataset(images[train_idx,,,], labels[train_idx,])
test_ds  <- dataset(images[test_idx,,,], labels[test_idx,])

train_dl <- dataloader(train_ds, batch_size = 16, shuffle = TRUE)
test_dl  <- dataloader(test_ds, batch_size = 16)

# Model
model <- nn_module(
  initialize = function() {
    self$conv1 <- nn_conv2d(3, 16, 3, padding = 1)
    self$conv2 <- nn_conv2d(16, 32, 3, padding = 1)
    self$fc1 <- nn_linear(32 * 16 * 16, 128)
    self$fc2 <- nn_linear(128, 4)
    self$pool <- nn_max_pool2d(2)
    self$relu <- nn_relu()
  },
  forward = function(x) {
    x %>% self$conv1() %>% self$relu() %>% self$pool() %>%
      self$conv2() %>% self$relu() %>% self$pool() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>% self$relu() %>%
      self$fc2()
  }
)

net <- model()
optimizer <- optim_adam(net$parameters, lr = 0.001)
loss_fn <- nn_mse_loss()

num_epochs <- 500
patience <- 10
best_loss <- Inf
epochs_no_improve <- 0

for (epoch in 1:num_epochs) {
  net$train()
  total_loss <- 0
  
  coro::loop(for (batch in train_dl) {
    optimizer$zero_grad()
    output <- net(batch$x)
    loss <- loss_fn(output, batch$y)
    loss$backward()
    optimizer$step()
    total_loss <- total_loss + loss$item()
  })
  
  # Validation
  net$eval()
  val_loss <- 0
  coro::loop(for (batch in test_dl) {
    output <- net(batch$x)
    loss <- loss_fn(output, batch$y)
    val_loss <- val_loss + loss$item()
  })
  
  cat(sprintf("Epoch %3d | Train Loss: %8.4f | Val Loss: %8.4f\n", epoch, total_loss, val_loss))
  
  # Early stopping
  if (val_loss < best_loss - 1e-4) {
    best_loss <- val_loss
    epochs_no_improve <- 0
    torch_save(net$state_dict(), "best_model.pt")
  } else {
    epochs_no_improve <- epochs_no_improve + 1
    if (epochs_no_improve >= patience) {
      cat(sprintf("Early stopping at epoch %d (no improvement for %d epochs).\n", epoch, patience))
      break
    }
  }
}

# Predict
net$eval()
preds <- net(images[test_idx,,,])$detach()


library(grid)
library(jpeg)

plot_pred <- function(i) {
  if (i > length(test_idx)) stop("Index exceeds test set size.")
  
  dataset_idx <- test_idx[i]
  img_id <- img_ids[dataset_idx]
  file <- sprintf("val2017/%012d.jpg", img_id)
  
  if (!file.exists(file)) stop("Image file not found: ", file)
  
  info <- img_info[img_info$id == img_id, ]
  img <- readJPEG(file)  # Use base JPEG reader instead of magick
  
  true <- as_array(labels[dataset_idx, ])
  pred <- as_array(net(images[dataset_idx,,,][NULL,,,])$squeeze(1))
  
  width <- info$width
  height <- info$height
  
  true_abs <- c(true[1] * width, true[2] * height, true[3] * width, true[4] * height)
  pred_abs <- c(pred[1] * width, pred[2] * height, pred[3] * width, pred[4] * height)
  
  grid.newpage()
  pushViewport(viewport(width = unit(1, "npc"), height = unit(1, "npc")))
  grid.raster(img, width = unit(1, "npc"), height = unit(1, "npc"))
  
  # Scale coordinates to NPC (normalized 0–1)
  grid.rect(x = unit(true_abs[1] / width, "npc"),
            y = unit(1 - true_abs[2] / height, "npc"),
            width = unit(true_abs[3] / width, "npc"),
            height = unit(true_abs[4] / height, "npc"),
            just = c("left", "top"),
            gp = gpar(col = "green", fill = NA, lwd = 3))
  
  grid.rect(x = unit(pred_abs[1] / width, "npc"),
            y = unit(1 - pred_abs[2] / height, "npc"),
            width = unit(pred_abs[3] / width, "npc"),
            height = unit(pred_abs[4] / height, "npc"),
            just = c("left", "top"),
            gp = gpar(col = "red", fill = NA, lwd = 3))
  
  grid.text("Ground Truth", x = unit(0.85, "npc"), y = unit(0.95, "npc"),
            gp = gpar(col = "green", fontsize = 12))
  grid.text("Prediction", x = unit(0.85, "npc"), y = unit(0.90, "npc"),
            gp = gpar(col = "red", fontsize = 12))
}


# Saving some predictions
png("prediction1.png", width = 640, height = 480)
plot_pred(1)
dev.off()

png("prediction2.png", width = 640, height = 480)
plot_pred(2)
dev.off()

png("prediction3.png", width = 640, height = 480)
plot_pred(3)
dev.off()
