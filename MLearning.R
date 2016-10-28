library("e1071")
library(class)
library(caret)
library(randomForest)
library(arules)
library(rpart)
library(ROCR)
library(ggplot2)
library(party)
library(adabag)
options(scipen=999)

#Read in source data from .csv file
import.csv <- function(filename) {
  return(read.csv(filename, sep = ",", header = TRUE))
}

filename<-"AvoxiData.csv"

my.data <- import.csv(filename)
my.data <- data.frame(my.data)

#Remove disqualified leads
my.data <- subset(my.data, Lead.Status != 'Disqualified')

#Subset dimensions of interest
vec1 <- c(5:11, 13, 16:17, 20:21, 23:24, 26, 29, 32, 35, 44)
my.data.sub <- my.data[,vec1]

#Set seed so that results can be replicated
set.seed(1000)

#########################################################################################################
#Universal Functions
#########################################################################################################

#Get_metrics function as defined in homework; takes as input vector of predicted and actual values from function call and returns data frame of metrics on accuracy of model
get_metrics<-function(pred,cutoff){
  count_acc<-0
  count_tnn<-0
  count_fnn<-0
  count_tnd<-0
  count_fnd<-0
  if(pred$prediction[1]!=0 ||pred$prediction[1]!=1){
    for (i in 1:nrow(pred)){
      if(pred$prediction[i]<=cutoff) {
        pred$prediction[i] = 0
      } else{
        pred$prediction[i] = 1
      }
    }
  }
  for(j in 1:nrow(pred)){
    if(pred[j,1]==pred[j,2]){
      count_acc <- count_acc+1
    }
    if(pred[j,1]==0&pred[j,2]==0){
      count_tnn <- count_tnn + 1
    }
    if(pred[j,1]==0&pred[j,2]==1){
      count_fnn <- count_fnn + 1
    }
    if(pred[j,2]==0){
      count_tnd<- count_tnd + 1
    } else{
      count_fnd<- count_fnd + 1
    }
  }
  #Col 1: True Negative Rate
  tnr<-count_tnn/count_tnd
  #Col 2: False Negative Rate
  fnr<-count_fnn/count_fnd
  #Col 3: Accuracy
  acc<-count_acc/nrow(pred)
  #Col 4: Precision
  prec<-count_tnn/(count_tnn+count_fnn)
  #Col 5: Recall (same as TNR)
  recall<-count_tnn/count_tnd
  outframe<-data.frame(tnr=tnr,fnr=fnr,acc=acc,precision=prec,recall=recall)
  return(outframe)
}

#Function 5: default predictor; returns vector with majority vote on training data output versus test data
get_pred_default<-function(train, test){
  nf <- ncol(train)
  pred<-vector()
  AUC <- as.numeric()
  colnames(train)[nf] <- "output"
  #train$output <- as.numeric(train$output)
  #train$output[train$output==2]<-0
  thresh<-mean(train$output)
  if(thresh>0.5){
    pred<-rep(1,nrow(test))
  } else{
    pred<-rep(0,nrow(test))
  }
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  
  predframe <- prediction(pred,test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  outvec<-data.frame(prediction=pred,true_output=test[,ncol(test)])
  return(outvec)
}


#########################################################################################################
#Decision Tree Algorithms
#########################################################################################################

do_cv_tree <- function(df, k, modstr, rpcontrol, ctcontrol) {
  #Randomize row entries of data frame
  set.seed(1000)
  dfrand <- df[sample(nrow(df)),]
  #Rows per fold
  numrows <- trunc(nrow(df)/k)
  startrow <- 1
  endrow <- numrows
  tnr<-0
  fnr<-0
  acc<-0
  prec<-0
  recall<-0
  aggout<-data.frame(tnr=tnr,fnr=fnr,acc=acc,precision=prec,recall=recall)
  #Return stats for each fold
  for (ii in 1:k){
    testframe <- dfrand[startrow:endrow,]
    trainframe <- dfrand[-(startrow:endrow),]
    
    #Determine type of model used by do_cv function call
    if(modstr=="dtree"){
      outmod <- get_pred_dtree(trainframe,testframe,rpcontrol)
    } else if(modstr=="ctree"){
      outmod <- get_pred_ctree(trainframe,testframe, ctcontrol)
    } else if(modstr=="adaboost"){
      outmod <- get_pred_adaboos(trainframe,testframe)
    } else if(modstr=="default"){
      outmod <- get_pred_default(trainframe,testframe)
    } else if(modstr=="rforest"){
      outmod <- get_pred_rforest(trainframe,testframe)
    } else{
      stop("Error - not a defined function type")
    }
    
    #Function call to get_metrics to return data frame of output measurements
    outstats<-get_metrics(outmod,0.5)
    aggout<-rbind(aggout,outstats)
    startrow <- startrow + numrows
    endrow <- min(nrow(dfrand),endrow + numrows)
  }
  aggout<-aggout[-1,]
  cat("The average accuracy for the k-fold models is: ",mean(aggout$acc),"\n")
  return(aggout)
}

get_pred_dtree<-function(train,test,control){
  nf <- ncol(train)
  pred<-vector()
  AUC <- as.numeric()
  colnames(train)[nf] <- "output"
  model <- rpart(output~.,data=train,method="class",control = rpart.control(cp = control))
  testframe<-test[,-ncol(test)]
  pred<-predict(model, testframe)
  pred <- as.data.frame(pred)
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  pred2 <- as.numeric()
  for(jj in 1:nrow(pred)){
    if(pred[jj,1]>=0.5){
      pred2[jj] <- names(pred)[1]
    } else {
      pred2[jj] <- names(pred)[2]
    }
  }
  
  predframe <- prediction(1-pred[[1]],test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  pred2 <- as.numeric(pred2)
  outvec<-data.frame(prediction=pred2,true_output=test[,ncol(test)])
  return(outvec)
}

get_pred_ctree<-function(train,test,control){
  nf <- ncol(train)
  pred<-vector()
  AUC <- as.numeric()
  colnames(train)[nf] <- "output"
  model <- ctree(output~.,data=train,controls=ctree_control(minsplit=control[1],minbucket=control[2],maxsurrogate=control[3]))
  testframe<-test[,-ncol(test)]
  predlist<-predict(model, testframe,type='prob',simply=FALSE)
  for(ll in 1:length(predlist)){
    pred[ll] <- predlist[[ll]][1]
  } 
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  pred2 <- as.numeric()
  for(jj in 1:length(pred)){
    if(pred[jj]>=0.5){
      pred2[jj] <- 1
    } else {
      pred2[jj] <- 0
    }
  }
  
  predframe <- prediction(pred2,test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  outvec<-data.frame(prediction=pred2,true_output=test[,ncol(test)])
  return(outvec)
}

get_pred_adaboos<-function(train,test){
  nf <- ncol(train)
  pred<-vector()
  AUC <- as.numeric()
  colnames(train)[nf] <- "output"
  train$output <- as.factor(train$output)
  model <- boosting(output~.,data=train,boost=TRUE,mfinal=15)
  testframe<-test[,-ncol(test)]
  pred<-predict(model, testframe)
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  pred2 <- as.numeric()
  
  pred <- as.numeric(pred$class)
  predframe <- prediction(pred,test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  outvec<-data.frame(prediction=pred,true_output=test[,ncol(test)])
  return(outvec)
}

get_pred_rforest<-function(train,test){
  nf <- ncol(train)
  pred<-vector()
  AUC <- as.numeric()
  colnames(train)[nf] <- "output"
  train$output <- as.factor(train$output)
  model <- randomForest(output~.,data=train,ntree=1000,importance=TRUE,proximity=TRUE)
  testframe<-test[,-ncol(test)]
  pred<-predict(model, testframe)
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  pred2 <- as.numeric()
  
  pred <- as.numeric(pred$class)
  predframe <- prediction(pred,test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  outvec<-data.frame(prediction=pred,true_output=test[,ncol(test)])
  return(outvec)
}

#########################################################################################################
#Numeric Input Algorithms - Requires Continuous Inputs (PCA)
#########################################################################################################

#PCA analysis
pcinput <- my.data.sub[,-ncol(my.data.sub)]
indx <- sapply(pcinput, is.factor)
pcinput[indx] <- lapply(pcinput[indx], function(x) as.numeric((x)))
pcinput[is.na(pcinput)] <- 0
PCAmodel <- prcomp(pcinput, retx=TRUE, center=TRUE, scale=TRUE)
screeplot(PCAmodel,50,type='lines')

outframe<-data.frame(PCANum=0,VarRet=0)
PCAvar<-(PCAmodel$sdev)^2
totalvar<-sum(PCAvar)

for (i in 1:length(PCAmodel$sdev)){
  sumvar<-0
  for(j in 1:i){
    sumvar <- sumvar + (PCAmodel$sdev[j])^2
  }
  per<-sumvar/totalvar
  vec<-c(i,per)
  outframe<-rbind(outframe,vec)
}
outframe<-outframe[-1,]
h1 <- ggplot(outframe, aes(x=PCANum,y=VarRet)) + geom_point(color="blue")+ggtitle("Sum Of Total Variance Retained Per PCA")+theme(plot.title = element_text(face="bold"))+ylim(0,1)
print(h1)

pcatransform <- as.data.frame(PCAmodel$x)
pcatransform <- cbind(pcatransform[1:15],my.data.sub$Won)
names(pcatransform)[ncol(pcatransform)] <- "output"

#Function 1: runs logistic regression on training data input and outputs data frame with prediction versus true value of test data
get_pred_logreg<-function(train, test){
  nf <- ncol(train)
  pred<-vector()
  colnames(train)[nf] <- "output"
  #train$output <- as.numeric(train$output)
  #train$output[train$output==2] <- 0
  model <- glm(output~.,data=train,family="binomial")
  testframe<-test[,-ncol(test)]
  pred<-predict(model, testframe, type="response")
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  
  outvec<-data.frame(prediction=pred,true_output=test[,ncol(test)])
  return(outvec)
}

#Function 2: runs SVM on training data input and outputs data frame with predicted and true values of test data
get_pred_svm<-function(train, test){
  nf <- ncol(train)
  pred<-vector()
  colnames(train)[nf] <- "output"
  #train$output <- as.numeric(train$output)
  #train$output[train$output==2]<-0
  model <- svm(output~.,data=train)
  testframe<-test[,-ncol(test)]
  pred<-predict(model, testframe)
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  
  predconv <- as.numeric(pred)
  predfinal<-vector()
  for(i in 1:length(predconv)){
    if(predconv[i]>=0.5){
      predfinal[i]=1
    } else{
      predfinal[i]=0
    }
  }
  
  predframe <- prediction(predfinal,test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  outvec<-data.frame(prediction=pred,true_output=test[,ncol(test)])
  return(outvec)
}

#Function 3: runs Naive Bayes model on training data input and outputs data frame with predicted and true values of the test data
get_pred_nb<-function(train, test){
  nf <- ncol(train)
  pred<-vector()
  colnames(train)[nf] <- "output"
  #train$output <- as.numeric(train$output)
  #train$output[train$output==2]<-0
  model <- naiveBayes(output~.,data=train)
  testframe<-test[,-ncol(test)]
  pred<-predict(model, testframe,type="raw")
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  
  predconv <- as.numeric(pred[,2])
  predfinal<-vector()
  for(i in 1:length(predconv)){
    if(predconv[i]>=0.5){
      predfinal[i]=1
    } else{
      predfinal[i]=0
    }
  }
  
  predframe <- prediction(predfinal,test[["output"]])
  perf <- performance(predframe, measure="tnr", x.measure="fnr")
  plot(perf, main="ROC Plot", col="blue")
  perf2 <- performance(predframe, measure="auc")
  AUC <- as.numeric(perf2@y.values)
  print(c('The AUC is ',round(AUC,2)))
  
  outvec<-data.frame(prediction=(as.numeric(pred[,2])),true_output=test[,ncol(test)])
  return(outvec)
}

#Function 4: runs K nearest neighbors model on training data input and outputs data frame with predicted and true values of the test data; requires input for "k" neighbors
get_pred_knn<-function(train, test, k){
  nf <- ncol(train)
  pred<-vector()
  colnames(train)[nf] <- "output"
  #train$output <- as.numeric(train$output)
  #train$output[train$output==2]<-0
  output<-train$output
  train1<-train[,-ncol(train)]
  test1<-test[,-ncol(test)]
  predvec <- knn(train1,test1,output,k,prob=FALSE)
  colnames(test)[nf] <- "output"
  #test$output <- as.numeric(test$output)
  #test$output[test$output==2]<-0
  outvec<-data.frame(prediction=(as.numeric(predvec)-1),true_output=test[,ncol(test)])
  return(outvec)
}

#Do_cv master function call requires data frame as input, "k" value for number of folds in cross validation, and an input for the type of model run in string format
#Valid inputs for modstr: {"svm","nb","knn" where k = integer, "logreg", and "default"}
#Example call would be do_cv(my.data,5,"5nn") for 5 nearest neighbors model with 5 fold cross validation
do_cv <- function(df, k, modstr) {
  #Randomize row entries of data frame
  set.seed(1000)
  dfrand <- df[sample(nrow(df)),]
  #Rows per fold
  numrows <- trunc(nrow(df)/k)
  startrow <- 1
  endrow <- numrows
  tnr<-0
  fnr<-0
  acc<-0
  prec<-0
  recall<-0
  aggout<-data.frame(tnr=tnr,fnr=fnr,acc=acc,precision=prec,recall=recall)
  #Return stats for each fold
  for (ii in 1:k){
    testframe <- dfrand[startrow:endrow,]
    trainframe <- dfrand[-(startrow:endrow),]
    
    #Determine type of model used by do_cv function call
    if(modstr=="logreg"){
      outmod <- get_pred_logreg(trainframe,testframe)
    } else if(modstr=="svm"){
      outmod <- get_pred_svm(trainframe,testframe)
    } else if (modstr=="nb"){
      outmod <- get_pred_nb(trainframe,testframe)
    } else if (modstr=="default"){
      outmod <- get_pred_default(trainframe,testframe)
    } else if (substr(modstr,nchar(modstr)-1,nchar(modstr))=="nn"){
      outmod <- get_pred_knn(trainframe,testframe,as.numeric(substr(modstr,1,nchar(modstr)-2)))
    } else{
      stop("Error - not a defined function type")
    }
    
    #Function call to get_metrics to return data frame of output measurements
    outstats<-get_metrics(outmod,0.5)
    aggout<-rbind(aggout,outstats)
    startrow <- startrow + numrows
    endrow <- min(nrow(dfrand),endrow + numrows)
  }
  aggout<-aggout[-1,]
  cat("The average accuracy for the k-fold models is: ",mean(aggout$acc),"\n")
  return(aggout)
}
