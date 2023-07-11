library(caret)

df <- data.frame(y = c(6, 8, 12, 14, 14, 15, 17, 22, 24, 23), x1 = c(2, 5, 4, 3, 4, 6, 7, 5, 8, 9), x2 = c(14, 12, 12, 13, 7, 8, 7, 4, 6, 5))
ctrl <- trainControl('cv', number = 5)
model <- train(y ~ x1 + x2, data = df, method = 'lm', trControl = ctrl)
y ~ x1 + x2