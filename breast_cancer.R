library(car)
library(caret)
library(dplyr)
library(ROCR)
library(e1071)
library(caret)
library(stats)
library(ggplot2)
library(DAAG)
library(party)
library(rpart)
library(mlbench)
library(pROC)
library(tree)
library(rpart.plot)

#install.packages("rpart.plot")

# pac <- c("DAAG", "party", "rpart", "mlbench", "pROC", "tree")
# 
# for (i in pac)
# {
#     install.packages(i)
# }
# 
# data <- subset(data, select = -c(X))
# 
# data$diagnosis <- as.factor(data$diagnosis)
# levels(data$diagnosis) <- c(0, 1)

data_scaled <- as.data.frame(scale(data[ , -which(names(data) == "diagnosis")]))
data_scaled$diagnosis <- data$diagnosis

pca <- prcomp(data[ , -which(names(data) == "diagnosis")], scale. = TRUE)
data_pca <- predict(pca, data[ , -which(names(data) == "diagnosis")])
data_pca <- as.data.frame(data_pca)

data_pca$diagnosis <- data$diagnosis

loading_scores <- as.data.frame(pca$rotation)
num_components <- 5

explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
cumulative_variance <- cumsum(explained_variance)

cat("Explained variance:\n")
print(explained_variance)
cat("Cumulative variance:\n")
print(cumulative_variance)

for (i in 1:num_components) {
  component_name <- paste("PC", i, sep = "")
  cat(paste(component_name, "cumulative variance:", cumulative_variance[i], "\n"))
}


important_pca <- data_pca[c("PC1", "PC2", "diagnosis")]

cor(data[c(3, 4, 5, 6, 7, 8, 9, 10)])

data_sub <- subset(data ,select = c(diagnosis, radius_mean, texture_mean, compactness_mean, concavity_mean))

set.seed(123)

test_data <- data_sub[sample(nrow(data_sub), round(0.2 * nrow(data_sub))), ]
train_data <- data_sub[sample(nrow(data_sub), round(0.8 * nrow(data_sub))), ]

#Logistic Regression

model_lr <- glm(diagnosis ~ ., data = train_data, family = "binomial")
summary(model_lr)

anova(model_lr, test = "Chisq")

predict_lr <- predict(model_lr, newdata = test_data, type = "response")
conf_matrix_lr <- table(test_data$diagnosis, ifelse(predict_lr > 0.5, "Yes", "No"))
print(conf_matrix_lr)

sum(diag(conf_matrix_lr))/sum(conf_matrix_lr)

pr_lr <- prediction(predict_lr, test_data$diagnosis)
prf_lr <- performance(pr_lr, measure = "tpr", x.measure = "fpr")

auc_lr <- performance(pr_lr, measure = "auc")
auc_lr <- auc_lr@y.values[[1]]
auc_lr

#Naive Bayes

model_nb <- naiveBayes(diagnosis ~ ., data = train_data, usepoisson = "TRUE")

predict_nb <- predict(model_nb, newdata = test_data)
conf_matrix_nb <- table(test_data$diagnosis, predict_nb)
print(conf_matrix_nb)

sum(diag(conf_matrix_nb))/sum(conf_matrix_nb)

pr_nb_prob <- predict(model_nb, test_data, type = "raw")[, 2]
pr_nb <- prediction(pr_nb_prob, test_data$diagnosis)
prf_nb <- performance(pr_nb, measure = "tpr", x.measure = "fpr")

auc_nb <- performance(pr_nb, measure = "auc")
auc_nb <- auc_nb@y.values[[1]]
auc_nb

#Tree

model_tree <- rpart(diagnosis ~ ., data = train_data)

printcp(model_tree)
plotcp(model_tree)

optimal_cp <- model_tree$cptable[which.min(model_tree$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(model_tree, cp = optimal_cp)

rpart.plot(pruned_tree)
print(pruned_tree)
summary(pruned_tree)

predict_tree <- predict(pruned_tree, newdata = test_data, type = "class")
predict_tree_prob <- predict(pruned_tree, newdata = test_data, type = "prob")
pr_tree <- prediction(predict_tree_prob[, 2], test_data$diagnosis)

conf_matrix_tree <- confusionMatrix(as.factor(predict_tree), as.factor(test_data$diagnosis))
print(conf_matrix_tree)

auc_tree <- performance(pr_tree, measure = "auc")
auc_value <- auc_tree@y.values[[1]]
auc_value

plot(prf_lr, col = "green", main = "Decision Tree, Logistic Regression and Naive Bayes ROC Curves\n Breast Cancer Prediction")
plot(performance(pr_tree, "tpr", "fpr"), col = "red", add = TRUE)
plot(performance(pr_nb, "tpr", "fpr"), col = "blue", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "naive Bayes"), col = c("green", "red", "blue"), lwd = 2)

