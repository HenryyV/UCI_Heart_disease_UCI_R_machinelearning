library(ggplot2)
library(dplyr)
heart=read.csv("heart.csv",header=TRUE)
names(heart) = c("age",   "sex",      "cp",       "trestbps", "chol",
                 "fbs",      "restecg",  "thalach",  "exang", "oldpeak",  "slope" ,  
                 "ca" ,      "thal"  ,   "target" )

#sex - (1 = male; 0 = female) 
#cp - chest pain type
#trestbps - resting blood pressure (in mm Hg
#chol - serum cholestoral in mg/dl 
#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
#restecg - resting electrocardiographic results
#thalach - maximum heart rate achieved 
#exang - exercise induced angina (1 = yes; 0 = no) 
#oldpeak - ST depression induced by exercise relative to rest
#slope - the slope of the peak exercise ST segment
# ca - number of major vessels (0-3) colored by flourosopy
# thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# target - have disease or not (1=yes, 0=no)


""" 1) LOOK AT THE DATA, Check your data visually, can you understand it? Can you easily spot valuable or
irrelevant information? Is it easy from a human point of view? A first assessment of the data can give
you a first intuition of wethere your problem is linearly separable or not. Generate visualizations and
comment any interesting features you encounter"""


ggplot(heart,aes(x=1,y= exang,color=target)) +geom_jitter() #When EXANG = 0, no angina, there are more targets
                                                            #more slope --> target 1
ggplot(heart,aes(x=1,y= sex,color=target)) +geom_jitter() #More targets 1 than 0 when male

ggplot(heart,aes(x=1,y= cp,color=target)) +geom_jitter() #More target 1 when cp 1,2 or 3

ggplot(heart,aes(x=1,y= oldpeak,color=target)) +geom_jitter() #Seems more target 1 in low oldpeak

ggplot(heart,aes(x=1,y= thalach,color=target)) +geom_jitter() #Higher talach -> target 1


ggplot(heart,aes(x=1,y= ca,color=target)) +geom_jitter() #lower ca -> target 1

ggplot(heart,aes(x=1,y= thal,color=target)) +geom_jitter() #thal 2 -> target 1

ggplot(heart,aes(x=thalach,y= slope,color=target)) +geom_jitter() #higher talach+slope -> target 1

heart2=read.csv("heart.csv",header=TRUE)
names(heart2) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                   "thalach","exang", "oldpeak","slope", "ca", "thal", "num")

#All this below is to create the graph of male/female disease/no disease

chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")
convert.magic <- function(obj,types){
  out <- lapply(1:length(obj),FUN = function(i){FUN1 <- 
    switch(types[i],
           character = as.character,
           numeric = as.numeric,
           factor = as.factor); FUN1(obj[,i])})
  names(out) <- colnames(obj)
  as.data.frame(out)
}

heart2 <- convert.magic(heart2,chclass)

levels(heart2$num) = c("No disease","Disease")
levels(heart2$sex) = c("female","male","")
mosaicplot(heart2$sex ~ heart2$num,
           main="Fate by Gender", shade=FALSE,color=TRUE,
           xlab="Gender", ylab="Heart disease")
#This graph shows the distribution of female and males and the disease. It is striking how much wemen have
#the disease compared with men.

""" After visualizing the data it is clear that there are variables that are useful and some that are not so
much. From a human point of view some characteristics can be seen  that are important but it is not easy to 
asses how much."""

""" 2) Think about your data splits and cross-validation strategy. How will the splits be generated? How shall
the final accuracy be computed? Basically, present your method in detail."""

library(caret)
set.seed(10)
inTrainRows <- createDataPartition(heart$target,p=0.75,list=FALSE)
train <- heart[inTrainRows,]
test <-  heart[-inTrainRows,]
nrow(train)/(nrow(test)+nrow(train))

""" We will divide our data into two splits which will be train and test with 75:25 to test all
the methods.""" 

""" 3) Generate PCA and LDA visualizations of your data, comment on the results"""
library(car)
scatterplotMatrix(train[2:6])

library(MASS)
train.lda <- lda(target~ ., data=train,scale=TRUE) #hacemos el lda, y te da los coeficientes de la combinacion lineal para predecir
train.lda

train.lda.values <- predict(train.lda)
ldahist(data = train.lda.values$x[,1], g=train.lda.values$class)
predictionslda=predict(train.lda,test)
predictionslda
table(predictionslda$class, test$target)
mean(predictionslda$class == test$target)


#PCA
pca <- prcomp(train, scale = TRUE)
pca$rotation
biplot(x = pca, scale = 0, cex = 0.6, col = c("blue4", "brown3")) #Mediante la función biplot() se puede obtener una representación bidimensional de las dos primeras componentes.

library(ggplot2)
pca$sdev^2
prop_varianza <- pca$sdev^2 / sum(pca$sdev^2)
prop_varianza
ggplot(data = data.frame(prop_varianza, pc = 1:14),
       aes(x = pc, y = prop_varianza)) +
  geom_col(width = 0.3) +
  scale_y_continuous(limits = c(0,1)) +
  theme_bw() +
  labs(x = "Componente principal",
       y = "Prop. de varianza explicada")

prop_varianza_acum <- cumsum(prop_varianza)
prop_varianza_acum
ggplot(data = data.frame(prop_varianza_acum, pc = 1:14),
       aes(x = pc, y = prop_varianza_acum, group = 1)) +
  geom_point() +
  geom_line() +
  theme_bw() +
  labs(x = "Componente principal",
       y = "Prop. varianza explicada acumulada")


#KNN
library(class)
model1<- knn(train=train[,1:13], test=test[,1:13],cl=train$target, k=20)
table(test$target, model1)
mean(test$target== model1)

"""
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
heart.new<- as.data.frame(lapply(heart[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14)],normalize))
head(heart.new)
library(caret)
set.seed(10)
inTrainRows2 <- createDataPartition(heart.new$target,p=0.75,list=FALSE)
train2 <- heart.new[inTrainRows2,]
test2 <-  heart[-inTrainRows2,]
nrow(train2)/(nrow(test2)+nrow(train2))

model2<- knn(train=train2[,1:13], test=test2[,1:13],cl=train2$target, k=20)
table(test2$target, model2)
mean(test2$target== model2)"""
#Si se hace normalizado da lo mismo


#NAIVE BAYES

library(naivebayes)
bayes <- naive_bayes(formula = as.factor(target) ~ca+exang+thalach+cp+oldpeak+slope,  data = train) #esto va un poco mejor que el knn
predictionsNB=predict(bayes,test)
mean(test$target==predictionsNB)

#SVM

library(e1071)
modelo_svm <- svm(as.factor(target)~., data = train, kernel = "linear", cost = 0.01, scale = TRUE)
predictionsvm=predict(modelo_svm,test)
mean(test$target==predictionsvm)
table(test$target,predictionsvm)

#calculamos el mejor cost con tune
set.seed(1)
svm_cv <- tune("svm", target~., data = train, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100, 150, 200)))
summary(svm_cv)
ggplot(data = svm_cv$performances, aes(x = cost, y = error)) + geom_line() +scale_x_continuous(limits=c(0,1))+ 
  geom_point() + labs(title = "Error de clasificación vs hiperparámetro C") + 
  theme_bw()

#ROCSVM
modelo_svm2 <- svm(as.factor(target)~., data = train, kernel = "linear", cost = 0.01, scale = TRUE,probability=TRUE)
x.svm.prob <- predict(modelo_svm2, type="prob", newdata=test, probability = TRUE)
require(ROCR)
x.svm.prob.rocr <- prediction(attr(x.svm.prob, "probabilities")[,1], test$target)
x.svm.perf <- performance(x.svm.prob.rocr, "tpr","fpr")
plot(x.svm.perf, col=4, main="SVM ROC curve")

#ROCKNN
model2<-  knn(train=train[,1:13], test=test[,1:13],cl=train$target, k=20,prob=TRUE)
prob <- attr(model2, "prob")
prob <- ifelse(model2 == "0", 1-prob, prob)
pred_knn <- prediction(prob, test$target)
pred_knn <- performance(pred_knn, "tpr", "fpr")
plot(pred_knn, avg= "threshold", lwd=1, main="kNN ROC curve")

#ROCLDA

library(ROCR)
pred <- prediction(predictionslda$posterior[,2], test$target) 
perf <- performance(pred,"tpr","fpr")
plot(perf)
