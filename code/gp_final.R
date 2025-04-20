# ====================== Environment Setup ======================
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(randomForest)
library(gbm)
library(glmnet)
set.seed(888)

# ====================== Data Preprocessing ======================
# Read data
setwd("C:/Users/Crystal/Desktop/5304/group project") # Replace with your data directory

data <- read.csv("bank-additional-full.csv", header = TRUE, sep = ";", stringsAsFactors = TRUE)

# Feature engineering
data$pdays_new <- as.factor(ifelse(data$pdays == 999, 0, 1))
data$pdays <- NULL

# Clean data (remove records containing "unknown")
data_clean <- data %>% 
  filter_all(all_vars(tolower(as.character(.)) != "unknown")) %>%
  mutate(
    y = factor(y, levels = c("no", "yes"))  # Explicitly set factor levels
  )

# ====================== Data Partitioning ======================
# Split data into three sets: 60% training, 20% validation, 20% test
trainIndex <- createDataPartition(data_clean$y, p = 0.6, list = FALSE)
remaining <- data_clean[-trainIndex,]
validIndex <- createDataPartition(remaining$y, p = 0.5, list = FALSE)

trainData <- data_clean[trainIndex,]
validData <- remaining[validIndex,]
testData <- remaining[-validIndex,]

# ====================== General Evaluation Function ======================
evaluate_model <- function(pred_prob, pred_class, actual) {
  # Ensure type consistency
  actual <- factor(actual, levels = c("no", "yes"))
  pred_class <- factor(pred_class, levels = c("no", "yes"))
  
  # Compute AUC (explicitly specify parameters)
  roc_obj <- roc(
    response = actual,
    predictor = pred_prob,
    levels = c("no", "yes"),
    direction = "<"
  )
  
  # Return comprehensive metrics
  list(
    Accuracy = mean(pred_class == actual),
    AUC = auc(roc_obj),
    MSE = mean((as.numeric(actual) - 1 - pred_prob)^2),
    SE = sd((as.numeric(actual) - 1 - pred_prob)^2)/sqrt(length(actual))
  )
}

# ====================== Logistic Regression ======================
logit_model <- glm(y ~ ., data = trainData, family = "binomial")

# Threshold optimization
valid_prob <- predict(logit_model, validData, type = "response")
roc_obj <- roc(validData$y, valid_prob, levels = c("no", "yes"), direction = "<")
best_threshold <- coords(roc_obj, "best", ret = "threshold")$threshold

# Test set prediction
test_prob_logit <- predict(logit_model, testData, type = "response")
test_pred_logit <- ifelse(test_prob_logit > best_threshold, "yes", "no") %>% 
  factor(levels = c("no", "yes"))

logit_metrics <- evaluate_model(test_prob_logit, test_pred_logit, testData$y)


### ====================== Lasso Regression ======================
# Parameter: lambda (alpha fixed as 1)
lasso_grid <- expand.grid(
  alpha = 1,  # Lasso fixes alpha = 1
  lambda = 10^seq(-4, 0, length=50)
)

best_lasso <- train(
  x = model.matrix(y ~ ., trainData)[,-1],
  y = trainData$y,
  method = "glmnet",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = lasso_grid,
  metric = "ROC"
)

# Test set prediction
x_test <- model.matrix(y ~ ., testData)[,-1]
test_prob_lasso <- predict(best_lasso, x_test, type = "prob")[,2]
test_pred_lasso <- ifelse(test_prob_lasso > 0.5, "yes", "no") %>% 
  factor(levels = c("no", "yes"))
lasso_metrics <- evaluate_model(test_prob_lasso, test_pred_lasso, testData$y)

### ====================== Ridge Regression ======================
# Parameter: lambda (alpha fixed as 0)
ridge_grid <- expand.grid(
  alpha = 0,  # Ridge fixes alpha = 0
  lambda = 10^seq(-4, 0, length=50)
)

best_ridge <- train(
  x = model.matrix(y ~ ., trainData)[,-1],
  y = trainData$y,
  method = "glmnet",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = ridge_grid,
  metric = "ROC"
)

# Test set prediction
test_prob_ridge <- predict(best_ridge, x_test, type = "prob")[,2]
test_pred_ridge <- ifelse(test_prob_ridge > 0.5, "yes", "no") %>% 
  factor(levels = c("no", "yes"))
ridge_metrics <- evaluate_model(test_prob_ridge, test_pred_ridge, testData$y)


### ====================== Revised Elastic Net ======================
# Data preparation (keep y as factor)
x_train <- model.matrix(y ~ ., trainData)[,-1]  # Remove the intercept column
y_train <- trainData$y  # Keep as factor, do not convert to numeric

# Check if the validation data contains both classes
print(table(y_train))  # Should show the sample sizes for "no" and "yes"

# Set proper training control parameters
en_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,       # Enable class probability calculation
  summaryFunction = twoClassSummary,  # Use classification evaluation metrics
  allowParallel = TRUE     # Enable parallel processing
)

# Parameter grid setup
en_grid <- expand.grid(
  alpha = seq(0, 1, 0.2),  # Mixing parameter: 0=Ridge, 1=Lasso
  lambda = 10^seq(-4, 0, length.out = 20)  # Regularization strength
)

# Train model (using classification settings)
best_en <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  family = "binomial",       # Explicitly specify binary classification
  trControl = en_ctrl,
  tuneGrid = en_grid,
  metric = "ROC"            # Optimize AUC
)

# Test set prediction
x_test <- model.matrix(y ~ ., testData)[,-1]
test_prob_en <- predict(best_en, x_test, type = "prob")[, "yes"]  # Get probability for "yes"
test_pred_en <- ifelse(test_prob_en > 0.5, "yes", "no") %>% 
  factor(levels = c("no", "yes"))

# Evaluation metrics
en_metrics <- evaluate_model(test_prob_en, test_pred_en, testData$y)


# ====================== Neural Network ======================

# Convert response y to binary 0/1 (nnet requires numeric type)
trainData$y_bin <- ifelse(trainData$y == "yes", 1, 0)
testData$y_bin  <- ifelse(testData$y == "yes", 1, 0)

# Construct formula: only use feature columns
feature_names <- setdiff(names(trainData), c("y", "y_bin", "y_num"))
formula_nn <- as.formula(paste("y_bin ~", paste(feature_names, collapse = " + ")))

# nnet does not accept factor variables, need dummy encoding
train_x <- model.matrix(formula_nn, data = trainData)[, -1]
test_x  <- model.matrix(formula_nn, data = testData)[, -1]

library(nnet)
set.seed(888)

nn_model <- nnet(
  x = train_x,
  y = trainData$y_bin,
  size = 5,         # Number of hidden layer nodes
  maxit = 200,      # Maximum iterations
  decay = 0.01,     # Weight decay (L2 regularization)
  linout = FALSE,   # Must be FALSE for classification problems
  trace = FALSE     # Do not print training process
)

# Predict probabilities
prob_pred <- predict(nn_model, newdata = test_x, type = "raw")

# Convert to classification labels
pred_class <- ifelse(prob_pred > 0.5, "yes", "no")
pred_class <- factor(pred_class, levels = c("yes", "no"))

# Compute confusion matrix
library(caret)
cm <- confusionMatrix(pred_class, testData$y, positive = "yes")
print(cm)

# Compute Mean Squared Error (based on probabilities)
mse <- mean((prob_pred - testData$y_bin)^2)
cat("MSE =", round(mse, 4), "\n")

library(pROC)

# Ensure testData$y is a factor with correct level order: negative "no", positive "yes"
testData$y <- factor(testData$y, levels = c("no", "yes"))

# Compute ROC object: explicitly set levels and direction
roc_obj <- roc(
  response = testData$y,
  predictor = as.vector(prob_pred), 
  levels = c("no", "yes"),
  direction = "<"
)

# Compute AUC
auc_val <- auc(roc_obj)
cat("AUC =", round(auc_val, 4), "\n")

nn_metrics <- evaluate_model(as.vector(prob_pred), pred_class, testData$y)




# ====================== Decision Tree ======================
# Correct parameter grid setup (only includes cp)
tree_grid <- expand.grid(cp = seq(0.0001, 0.02, length.out = 20))

# Set additional parameters via rpart.control
tree_control <- rpart.control(
  minsplit = 10,   # Minimum number of observations to attempt a split
  maxdepth = 15,   # Maximum tree depth
  minbucket = 20   # Minimum number of samples in terminal nodes
)

best_tree <- train(
  y ~ .,
  data = trainData,
  method = "rpart",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = tree_grid,
  metric = "ROC",
  control = tree_control  # Pass fixed parameters
)

# Test set prediction
test_prob_tree <- predict(best_tree, testData, type = "prob")[,2]
test_pred_tree <- predict(best_tree, testData)
tree_metrics <- evaluate_model(test_prob_tree, test_pred_tree, testData$y)

# ====================== Random Forest ======================
# Parameter optimization
rf_grid <- expand.grid(
  mtry = c(3, 5, 7),
  splitrule = "gini",
  min.node.size = c(5, 10)
)

best_rf <- train(
  y ~ .,
  data = trainData,
  method = "ranger",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = rf_grid,
  importance = "impurity",
  metric = "ROC"
)

# Prediction and evaluation
test_prob_rf <- predict(best_rf, testData, type = "prob")[,2]
test_pred_rf <- predict(best_rf, testData)
rf_metrics <- evaluate_model(test_prob_rf, test_pred_rf, testData$y)


### ====================== Bagging ======================
library(ipred)
library(caret)
library(rpart)
library(pROC)
library(doParallel)

# Assume trainData and testData are already loaded
# Ensure the response variable y is a factor with levels ordered ("no" for control, "yes" for case)
trainData$y <- factor(trainData$y, levels = c("no", "yes"))
testData$y <- factor(testData$y, levels = c("no", "yes"))
# Start parallel computing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Set parameter grid
param_grid <- expand.grid(
  nbagg = c(10, 25, 50),
  minsplit = c(10, 20),
  maxdepth = c(5, 10),
  cp = c(0.001, 0.01)
)

# Initialize results
results <- data.frame()

# Loop through parameter combinations
for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]
  
  # Build control parameters
  ctrl <- rpart.control(
    minsplit = params$minsplit,
    maxdepth = params$maxdepth,
    cp = params$cp
  )
  
  # Train model
  model <- bagging(
    y ~ ., 
    data = trainData, 
    nbagg = params$nbagg,
    coob = TRUE,
    control = ctrl
  )
  
  # Prediction
  prob <- predict(model, testData, type = "prob")[, 2]
  pred <- predict(model, testData, type = "class")
  
  # Compute AUC
  auc_val <- auc(testData$y, prob)
  
  # Save results
  results <- rbind(results, data.frame(
    nbagg = params$nbagg,
    minsplit = params$minsplit,
    maxdepth = params$maxdepth,
    cp = params$cp,
    AUC = auc_val
  ))
}

# Stop parallel computing
stopCluster(cl)

# Identify best parameter combination
best_params <- results[which.max(results$AUC), ]
print(best_params)

# Re-train model using best parameters
best_ctrl <- rpart.control(
  minsplit = best_params$minsplit,
  maxdepth = best_params$maxdepth,
  cp = best_params$cp
)

bagging_model <- bagging(
  y ~ ., 
  data = trainData, 
  nbagg = best_params$nbagg,
  control = best_ctrl
)

# Prediction and evaluation
test_prob_bagging <- predict(bagging_model, testData, type = "prob")[, 2]
test_pred_bagging  <- predict(bagging_model, testData, type = "class")
bag_metrics <- evaluate_model(test_prob_bagging, test_pred_bagging, testData$y)



# ====================== GBM ======================
# Set parameter grid
gbm_grid <- expand.grid(
  n.trees = c(100, 200, 500),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.001, 0.01, 0.1),
  n.minobsinnode = c(5, 10, 20)
)

# Train GBM model
best_gbm <- train(
  y ~ .,
  data = trainData,
  method = "gbm",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneGrid = gbm_grid,
  verbose = FALSE,
  metric = "ROC"
)

# Print best parameter combination
print(best_gbm$bestTune)

# Prediction and evaluation on test set
test_prob_gbm <- predict(best_gbm, testData, type = "prob")[,2]
test_pred_gbm <- predict(best_gbm, testData)
gbm_metrics <- evaluate_model(test_prob_gbm, test_pred_gbm, testData$y)

# Print evaluation metrics
print(gbm_metrics)



## Variable Importance
# ====================== GBM Variable Importance Visualization ======================
library(ggplot2)
library(dplyr)

# Extract and prepare importance data
gbm_importance <- varImp(best_gbm, scale = TRUE)
gbm_imp_df <- gbm_importance$importance %>%
  mutate(Feature = rownames(.)) %>%
  arrange(desc(Overall)) %>%
  head(15)
print(gbm_imp_df )

# Create formatted labels if needed
# gbm_imp_df$Feature <- gsub("_", " ", gbm_imp_df$Feature)  # Convert underscores to spaces
# gbm_imp_df$Feature <- tools::toTitleCase(gbm_imp_df$Feature)  # Title case formatting

# Generate visualization
ggplot(gbm_imp_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "#0072B2", width = 0.8) +  # Use academic blue color
  coord_flip() +
  labs(
    title = "Variable Importance in GBM Model",
    subtitle = "Top 15 Predictive Features",
    x = "Feature",
    y = "Importance Score (Scaled)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = unit(c(1,1,1,1), "cm")
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)))


# ====================== Ensemble Learning Enhancement (using limSolve package) ====================== 

# ---------------------- 0. Load Required Packages ----------------------
if (!require("limSolve")) install.packages("limSolve")
library(limSolve)
library(hdm)
library(pROC)

# ====================== Data Preparation Stage ======================

# ---------------------- 1. Construct Validation Set Prediction Probability Matrix ----------------------
valid_probs <- data.frame(
  logit  = predict(logit_model, validData, type = "response"),
  tree   = predict(best_tree, validData, type = "prob")[, "yes"],
  rf     = predict(best_rf, validData, type = "prob")[, "yes"],
  gbm    = predict(best_gbm, validData, type = "prob")[, "yes"],
  bag    = predict(final_model, validData, type = "prob")[, "yes"],
  nn     = as.vector(predict(nn_model, newdata = model.matrix(y ~ ., validData)[, -1])),
  lasso  = predict(best_lasso, model.matrix(y ~ ., validData)[, -1], type = "prob")[, "yes"],
  ridge  = predict(best_ridge, model.matrix(y ~ ., validData)[, -1], type = "prob")[, "yes"]
)
X_valid <- as.matrix(valid_probs)
y_valid <- ifelse(validData$y == "yes", 1, 0)  # Convert to numeric

# ---------------------- 2. Construct Test Set Prediction Probability Matrix ----------------------
test_probs <- data.frame(
  logit  = test_prob_logit,
  tree   = test_prob_tree,
  rf     = test_prob_rf,
  gbm    = test_prob_gbm,
  bag    = final_prob,
  nn     = as.vector(prob_pred),
  lasso  = test_prob_lasso,
  ridge  = test_prob_ridge
)
X_test <- as.matrix(test_probs)

# ====================== Weight Calculation Module ======================

# ---------------------- Method 1: GR Weighting (Constrained, No Intercept) -----------------------
gr_unres_noconst <- limSolve::lsei(
  A = X_valid, 
  B = y_valid, 
  E = matrix(1, nrow = 1, ncol = ncol(X_valid)),  # sum(weights) = 1
  F = 1,
  G = diag(ncol(X_valid)),                        # weights >= 0
  H = rep(0, ncol(X_valid))
)
weights_gr_unres_noconst <- gr_unres_noconst$X

# ---------------------- Method 2: GR Weighting (Constrained, With Intercept) -----------------------
X_valid_ext <- cbind(1, X_valid)
temp_mat <- diag(ncol(X_valid_ext))
temp_mat[1, 1] <- 0  

gr_unres_const <- limSolve::lsei(
  A = X_valid_ext, 
  B = y_valid, 
  E = matrix(c(0, rep(1, ncol(X_valid))), nrow = 1),  # Model coefficients sum to 1
  F = 1,
  G = temp_mat,                                   
  H = rep(0, ncol(X_valid_ext))
)
weights_gr_unres_const <- gr_unres_const$X

# ---------------------- Method 3: GR (Unconstrained, No Intercept) ---------------------
gr_res_noconst <- limSolve::lsei(
  A = X_valid, 
  B = y_valid,
  E = NULL,  # No equality constraints
  F = NULL,
  G = NULL,  # No inequality constraints
  H = NULL
)
weights_gr_res_noconst <- gr_res_noconst$X

# ---------------------- Method 4: GR Weighting (Unconstrained, With Intercept) -----------------------
X_valid_const <- cbind(1, X_valid)
gr_res_const <- limSolve::lsei(
  A = X_valid_const, 
  B = y_valid,
  E = NULL,  # No constraints
  F = NULL,
  G = NULL,  
  H = NULL
)
weights_gr_res_const  <- gr_res_const$X

# ---------------------- Method 5: LASSO Weighting (No Intercept) -----------------------
lasso_comb <- rlasso(X_valid, y_valid, post = FALSE)
weights_lasso <- as.numeric(coef(lasso_comb))
if(length(weights_lasso) == (ncol(X_valid) + 1)) {
  weights_lasso <- weights_lasso[-1]  # Remove intercept term
}

# ---------------------- Method 6: LASSO Weighting (With Intercept) ---------------------
X_valid_ext_lasso <- cbind(1, X_valid)
n_models <- ncol(X_valid)
E_lasso_ext <- matrix(c(0, rep(1, n_models)), nrow = 1)
G_lasso_ext <- cbind(0, diag(n_models))  
lasso_const <- limSolve::lsei(
  A = X_valid_ext_lasso, 
  B = y_valid, 
  E = E_lasso_ext,
  F = 1,
  G = G_lasso_ext,
  H = rep(0, n_models)
)
weights_lasso_const <- lasso_const$X

# ====================== Ensemble Prediction Module ======================

# ---------------------- Validation Set Ensemble Prediction ----------------------
valid_ensemble_probs <- list(
  gr_unres_noconst = as.vector(X_valid %*% weights_gr_unres_noconst),
  gr_unres_const = as.vector(X_valid_ext %*% weights_gr_unres_const),
  gr_res_noconst = as.vector(X_valid %*% weights_gr_res_noconst),        # Unconstrained, no intercept
  gr_res_const = as.vector(X_valid_const %*% weights_gr_res_const),
  lasso = as.vector(X_valid %*% weights_lasso),
  lasso_const = as.vector(X_valid_ext_lasso %*% weights_lasso_const)  # New
)

# ---------------------- Test Set Ensemble Prediction ----------------------
test_ensemble_probs <- list(
  gr_unres_noconst = as.vector(X_test %*% weights_gr_unres_noconst),
  gr_unres_const = as.vector(cbind(1, X_test) %*% weights_gr_unres_const),
  gr_res_noconst = as.vector(X_test %*% weights_gr_res_noconst),         # Unconstrained, no intercept
  gr_res_const = as.vector(cbind(1, X_test) %*% weights_gr_res_const),
  lasso = as.vector(X_test %*% weights_lasso),
  lasso_const = as.vector(cbind(1, X_test) %*% weights_lasso_const)  # New
)

# ====================== Evaluation Module ======================

# ---------------------- Define Evaluation Function ----------------------
evaluate_metrics <- function(prob, actual) {
  actual_num <- ifelse(actual == "yes", 1, 0)
  
  if(length(unique(prob)) < 2) {
    auc_val <- NA
  } else {
    roc_obj <- roc(response = actual_num, predictor = prob, quiet = TRUE)
    auc_val <- as.numeric(auc(roc_obj))
  }
  
  mse_val <- mean((actual_num - prob)^2)
  se_val <- sd((actual_num - prob)^2) / sqrt(length(actual_num))
  
  return(list(AUC = auc_val, MSE = mse_val, SE = se_val))
}

# ---------------------- Compute Results ----------------------
# Ensemble model results
metrics_list <- lapply(test_ensemble_probs, evaluate_metrics, actual = testData$y)

# Individual model results
individual_metrics <- list(
  logit = evaluate_model(test_prob_logit, test_pred_logit, testData$y),
  tree = evaluate_model(test_prob_tree, test_pred_tree, testData$y),
  rf = evaluate_model(test_prob_rf, test_pred_rf, testData$y),
  gbm = evaluate_model(test_prob_gbm, test_pred_gbm, testData$y),
  bag = evaluate_model(test_prob_bagging, test_pred_bagging, testData$y),
  nn = evaluate_model(as.vector(prob_pred), pred_class, testData$y),
  lasso = evaluate_model(test_prob_lasso, test_pred_lasso, testData$y),
  ridge = evaluate_model(test_prob_ridge, test_pred_ridge, testData$y)
)

# ====================== Results Output Module ======================

# ---------------------- Output Weight Information ----------------------
names(weights_gr_unres_noconst) <- colnames(X_valid)
names(weights_gr_unres_const)   <- c("Intercept", colnames(X_valid))
names(weights_gr_res_noconst)   <- colnames(X_valid)
names(weights_gr_res_const)     <- c("Intercept", colnames(X_valid))
names(weights_lasso)            <- colnames(X_valid)
names(weights_lasso_const)      <- c("Intercept", colnames(X_valid))

# ---------------------- Build and Output Weight Matrix ----------------------
base_models <- colnames(X_valid)

weight_matrix <- data.frame(
  Model = c("Intercept", base_models),
  GR_Unres_Noconst      = c(0, round(weights_gr_unres_noconst, 4)),  # Method 1
  GR_Unres_Const        = round(weights_gr_unres_const, 4),          # Method 2
  GR_Res_Noconst        = c(0, round(weights_gr_res_noconst, 4)),    # Method 3
  GR_Res_Const          = round(weights_gr_res_const, 4),            # Method 4
  LASSO_Noconst         = c(0, round(weights_lasso, 4)),             # Method 5
  LASSO_Const           = round(weights_lasso_const, 4)              # Method 6
)

cat("\n========== Ensemble Model Weight Matrix ==========\n")
print(weight_matrix)

# ---------------------- Output Model Evaluation Results ----------------------

# Individual model results table
indiv_table <- data.frame(
  Model = names(individual_metrics),
  AUC = sapply(individual_metrics, function(x) round(x$AUC, 6)),
  MSE = sapply(individual_metrics, function(x) round(x$MSE, 6)),
  SE = sapply(individual_metrics, function(x) round(x$SE, 6)),
  Type = "Individual"
)

# Ensemble model results table
ensemble_table <- data.frame(
  Model = names(metrics_list),
  AUC = sapply(metrics_list, function(x) round(x$AUC, 6)),
  MSE = sapply(metrics_list, function(x) round(x$MSE, 6)),
  SE = sapply(metrics_list, function(x) round(x$SE, 6)),
  Type = "Ensemble"
)

# ---------------------- Output Combined Sorted Results (Sorted by AUC Descending) ----------------------
final_results <- rbind(ensemble_table, indiv_table)
final_results <- final_results[order(-final_results$AUC), ]

cat("\n========== All Models Performance Ranking (Sorted by Descending AUC) ==========\n")
print(final_results)
