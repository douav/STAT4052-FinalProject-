# Doua Vue 
# STAT 4052 
# Final Project
# 05/08/2026



## Data 
```{r}
library(glmnet)
library(randomForest)
library(caret)
library(MASS)
 
set.seed(123)

data <- read.csv("C:/Users/Doua Vue/OneDrive/Documents/STAT4052/processed.cleveland.data",
                 header = FALSE, na.strings = "?",
                 col.names = c("age", "sex", "cp", "trestbps", "chol", "fbs",
                               "restecg", "thalach", "exang", "oldpeak", "slope",
                               "ca", "thal", "target"))
data <- na.omit(data)
 
bc <- boxcox(thalach ~ age + sex + cp + trestbps + chol + fbs + restecg + exang + 
               oldpeak + slope + ca + thal, data = data)
bc$x[which.max(bc$y)]  

```

## Train Test Split 
```{r}
idx <- createDataPartition(data$thalach, p = 0.80, list = FALSE)
train <- data[idx, ]
test <- data[-idx, ]
 
f <- (thalach ~ age + sex + cp + trestbps + chol + fbs + restecg + exang + oldpeak 
      + slope + ca + thal)
 
x_train <- model.matrix(f, train)[, -1]
x_test <- model.matrix(f, test)[, -1]
y_train <- train$thalach
y_test <- test$thalach

```

## Calcuations 
```{r}
calc <- function(actual, pred) {
  e <- actual - pred
  c(MSE = mean(e^2),
    RMSE = sqrt(mean(e^2)),
    MAE = mean(abs(e)),
    R2 = 1 - sum(e^2) / sum((actual - mean(actual))^2))
}

```

## Multiple Linear Regression 
```{r}
mod <- lm(f, data = train)
summary(mod)

par(mfrow = c(2,2))
plot(mod)

par(mfrow = c(1,1))
mod_calc <- calc(y_test, predict(mod, test))
mod_calc

```

## Ridge 
```{r}
ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)
plot(ridge)

ridge_calc  <- calc(y_test, as.vector(predict(ridge, x_test, s = "lambda.min")))
ridge_calc

```

## Lasso 
```{r}
lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10)
plot(lasso)

coef(lasso, s = "lambda.min")
lasso_calc<- calc(y_test, as.vector(predict(lasso, x_test, s = "lambda.min")))
lasso_calc

```

## Random Forest 
```{r}
rf <- randomForest(f, data = train, ntree = 500, importance = TRUE)

varImpPlot(rf, type = 1, main = "Variable Importance (%IncMSE)", 
           col = colorRampPalette(c("blue","red"))(12))
rf_calc <- calc(y_test, predict(rf, test))
rf_calc
```

## Comparison 
```{r}
results <- rbind(MLR = mod_calc, Ridge = ridge_calc, Lasso = lasso_calc, RF = rf_calc)
round(results, 3)

```

## Predicted vs Actual 
```{r}
# ── PREDICTED VS ACTUAL (best model) ─────────────────────────
plot(y_test, predict(rf, test), pch = 16, col = "steelblue",
     xlab = "Actual", ylab = "Predicted", main = "RF: Predicted vs Actual")
abline(0, 1, col = "red", lwd = 2)

```
