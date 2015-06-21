# Human Activity Recognition
Beatriz Ortiz  
20 Jun 2015  


## Overview:

 In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. In this report that we will create later, we try to investigate "how (well)" an activity was performed by the wearer. 


## Prepare the Enviroment

First, we are going to load the necesary libraries to generate code and plots and prepare the enviroment to use Knit options.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(knitr)
opts_chunk$set(echo = TRUE, results = 'hold')
```


```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```
## DownLoad data. 

Now we are going download data read csv files into training and test. Then we will go to prepprocess data using cross validation


```r
if (!file.exists("./har")){ 
        dir.create("./har")
        url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url,destfile="./har/pml-training.csv",method="curl")
download.file(url,destfile="./har/pml-testing.csv",method="curl")
}

training <- read.csv("./har/pml-training.csv")
test <- read.csv("./har/pml-testing.csv")
```

### Spliting the trainig set 

Now we will use the function createDataPartition to create balanced splits of the trainig set. We will create a 60/40% split of the data. Previously we set a seed.


```r
set.seed(8420)
trainIndex <- createDataPartition(training$classe, p = 0.6,  list = FALSE)
train_part <- training[trainIndex, ]
test_part  <- training[-trainIndex, ]
dim(train_part)
dim(test_part)
```

```
## [1] 11776   160
## [1] 7846  160
```

## Cleaning and pre-Processing Data

### Near Zero Covariate

Prior to modeling, we want to identify and eliminate columns with a few unique numeric values. To do this, we use the neaZeroValues function:


```r
nzv <- nearZeroVar(train_part , saveMetrics= TRUE)
nzv <- nearZeroVar(train_part)
train_part <- train_part[, -nzv]
```

Now, we want to know in which columns there are missing values and the total number in each column. We will remove those, whose values are mostly NA 


```r
 colSums(is.na(train_part))
```

```
##                        X                user_name     raw_timestamp_part_1 
##                        0                        0                        0 
##     raw_timestamp_part_2           cvtd_timestamp               num_window 
##                        0                        0                        0 
##                roll_belt               pitch_belt                 yaw_belt 
##                        0                        0                        0 
##         total_accel_belt            max_roll_belt           max_picth_belt 
##                        0                    11525                    11525 
##            min_roll_belt           min_pitch_belt      amplitude_roll_belt 
##                    11525                    11525                    11525 
##     amplitude_pitch_belt     var_total_accel_belt            avg_roll_belt 
##                    11525                    11525                    11525 
##         stddev_roll_belt            var_roll_belt           avg_pitch_belt 
##                    11525                    11525                    11525 
##        stddev_pitch_belt           var_pitch_belt             avg_yaw_belt 
##                    11525                    11525                    11525 
##          stddev_yaw_belt             var_yaw_belt             gyros_belt_x 
##                    11525                    11525                        0 
##             gyros_belt_y             gyros_belt_z             accel_belt_x 
##                        0                        0                        0 
##             accel_belt_y             accel_belt_z            magnet_belt_x 
##                        0                        0                        0 
##            magnet_belt_y            magnet_belt_z                 roll_arm 
##                        0                        0                        0 
##                pitch_arm                  yaw_arm          total_accel_arm 
##                        0                        0                        0 
##            var_accel_arm              gyros_arm_x              gyros_arm_y 
##                    11525                        0                        0 
##              gyros_arm_z              accel_arm_x              accel_arm_y 
##                        0                        0                        0 
##              accel_arm_z             magnet_arm_x             magnet_arm_y 
##                        0                        0                        0 
##             magnet_arm_z             max_roll_arm            max_picth_arm 
##                        0                    11525                    11525 
##              max_yaw_arm             min_roll_arm            min_pitch_arm 
##                    11525                    11525                    11525 
##              min_yaw_arm       amplitude_roll_arm      amplitude_pitch_arm 
##                    11525                    11525                    11525 
##        amplitude_yaw_arm            roll_dumbbell           pitch_dumbbell 
##                    11525                        0                        0 
##             yaw_dumbbell        max_roll_dumbbell       max_picth_dumbbell 
##                        0                    11525                    11525 
##        min_roll_dumbbell       min_pitch_dumbbell  amplitude_roll_dumbbell 
##                    11525                    11525                    11525 
## amplitude_pitch_dumbbell     total_accel_dumbbell       var_accel_dumbbell 
##                    11525                        0                    11525 
##        avg_roll_dumbbell     stddev_roll_dumbbell        var_roll_dumbbell 
##                    11525                    11525                    11525 
##       avg_pitch_dumbbell    stddev_pitch_dumbbell       var_pitch_dumbbell 
##                    11525                    11525                    11525 
##         avg_yaw_dumbbell      stddev_yaw_dumbbell         var_yaw_dumbbell 
##                    11525                    11525                    11525 
##         gyros_dumbbell_x         gyros_dumbbell_y         gyros_dumbbell_z 
##                        0                        0                        0 
##         accel_dumbbell_x         accel_dumbbell_y         accel_dumbbell_z 
##                        0                        0                        0 
##        magnet_dumbbell_x        magnet_dumbbell_y        magnet_dumbbell_z 
##                        0                        0                        0 
##             roll_forearm            pitch_forearm              yaw_forearm 
##                        0                        0                        0 
##         max_roll_forearm        max_picth_forearm         min_roll_forearm 
##                    11525                    11525                    11525 
##        min_pitch_forearm  amplitude_pitch_forearm      total_accel_forearm 
##                    11525                    11525                        0 
##        var_accel_forearm          gyros_forearm_x          gyros_forearm_y 
##                    11525                        0                        0 
##          gyros_forearm_z          accel_forearm_x          accel_forearm_y 
##                        0                        0                        0 
##          accel_forearm_z         magnet_forearm_x         magnet_forearm_y 
##                        0                        0                        0 
##         magnet_forearm_z                   classe 
##                        0                        0
```
As we can see, all the columns with missing values, have more than a 90% of this NA. I will go to remove then.


```r
 train_part <- train_part[ ,(colSums(is.na(train_part)) == 0)]
```

Also we remove columns like X(id number), user_name, all timestamp and numwindow. We don't need this columns for prediction.


```r
 train_part <- train_part[ ,-c(1:5)]
 str(train_part)
```

```
## 'data.frame':	11776 obs. of  54 variables:
##  $ num_window          : int  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.42 1.48 1.45 1.42 1.42 1.43 1.45 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.06 8.09 8.13 8.16 8.18 8.18 8.2 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0.02 0 0.02 0.02 0.02 0.02 0.02 0.03 0.02 0 ...
##  $ gyros_belt_y        : num  0 0 0.02 0 0 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -22 -20 -21 -21 -22 -22 -20 -21 -22 -21 ...
##  $ accel_belt_y        : int  4 5 2 4 3 4 2 2 2 2 ...
##  $ accel_belt_z        : int  22 23 24 21 21 21 24 23 23 22 ...
##  $ magnet_belt_x       : int  -7 -2 -6 0 -4 -2 1 -5 -2 -1 ...
##  $ magnet_belt_y       : int  608 600 600 603 599 603 602 596 602 597 ...
##  $ magnet_belt_z       : int  -311 -305 -302 -312 -311 -313 -312 -317 -319 -310 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -129 ...
##  $ pitch_arm           : num  22.5 22.5 22.1 22 21.9 21.8 21.7 21.5 21.5 21.4 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0.02 0.02 0 0.02 0 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  -0.02 -0.02 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 0 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 0 0 0 0 -0.02 0 0 -0.03 ...
##  $ accel_arm_x         : int  -290 -289 -289 -289 -289 -289 -288 -290 -288 -289 ...
##  $ accel_arm_y         : int  110 110 111 111 111 111 109 110 111 111 ...
##  $ accel_arm_z         : int  -125 -126 -123 -122 -125 -124 -122 -123 -123 -124 ...
##  $ magnet_arm_x        : int  -369 -368 -374 -369 -373 -372 -369 -366 -363 -374 ...
##  $ magnet_arm_y        : int  337 344 337 342 336 338 341 339 343 342 ...
##  $ magnet_arm_z        : int  513 513 506 513 509 510 518 509 520 510 ...
##  $ roll_dumbbell       : num  13.1 12.9 13.4 13.4 13.1 ...
##  $ pitch_dumbbell      : num  -70.6 -70.3 -70.4 -70.8 -70.2 ...
##  $ yaw_dumbbell        : num  -84.7 -85.1 -84.9 -84.5 -85.1 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -233 -232 -233 -234 -232 -234 -232 -233 -233 -234 ...
##  $ accel_dumbbell_y    : int  47 46 48 48 47 46 47 47 47 47 ...
##  $ accel_dumbbell_z    : int  -269 -270 -270 -269 -270 -272 -269 -269 -270 -270 ...
##  $ magnet_dumbbell_x   : int  -555 -561 -554 -558 -551 -555 -549 -564 -554 -554 ...
##  $ magnet_dumbbell_y   : int  296 298 292 294 295 300 292 299 291 294 ...
##  $ magnet_dumbbell_z   : num  -64 -63 -68 -66 -70 -74 -65 -64 -65 -63 ...
##  $ roll_forearm        : num  28.3 28.3 28 27.9 27.9 27.8 27.7 27.6 27.5 27.2 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 -63.8 -63.9 ...
##  $ yaw_forearm         : num  -153 -152 -152 -152 -152 -152 -152 -152 -152 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.02 0.03 0.02 0.02 0.02 0.02 0.03 0.02 0.02 0 ...
##  $ gyros_forearm_y     : num  0 -0.02 0 -0.02 0 -0.02 0 -0.02 0.02 -0.02 ...
##  $ gyros_forearm_z     : num  -0.02 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 -0.03 -0.02 ...
##  $ accel_forearm_x     : int  192 196 189 193 195 193 193 193 191 192 ...
##  $ accel_forearm_y     : int  203 204 206 203 205 205 204 205 203 201 ...
##  $ accel_forearm_z     : int  -216 -213 -214 -215 -215 -213 -214 -214 -215 -214 ...
##  $ magnet_forearm_x    : int  -18 -18 -17 -9 -18 -9 -16 -17 -11 -16 ...
##  $ magnet_forearm_y    : num  661 658 655 660 659 660 653 657 657 656 ...
##  $ magnet_forearm_z    : num  473 469 473 478 470 474 476 465 478 472 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```


Clean now the test_part and test set

```r
trcolnames <- colnames(train_part)
trcolnames2 <- colnames(train_part[ ,-54])
        
test_part <- test_part[trcolnames]
test <- test[trcolnames2]
```

## Cross Validation
I will use trainControl functionc to specifiy the type of resampling.  I will specifie 10-fold repeated cross-validation with repeated 3 times. Then I will pass this value directly to the train function as an argument .



```r
fitControl <- trainControl(method = "repeatedcv",
                            number = 10,
                           repeats = 3)
```
### Modeling 

Now we are going to fit a model using classe variable as outcome value. First I use rpart method. We pass fitControl as argument. 


```r
modelFit <- train(classe ~ ., data = train_part,
                  method = "rpart",
                 trControl = fitControl)
print(modelFit)
```

```
## CART 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## 
## Summary of sample sizes: 10599, 10599, 10597, 10598, 10599, 10599, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03945183  0.5244863  0.38984214  0.03046431   0.04864015
##   0.05908875  0.4277555  0.22882576  0.06116588   0.10359572
##   0.11675368  0.3348298  0.07699313  0.03929196   0.05975369
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03945183.
```

And this is the Plot for modelFit

```r
fancyRpartPlot(modelFit$finalModel)
```

![](HAR_files/figure-html/unnamed-chunk-12-1.png) 

Now we are going to predict against to the test_part set.  we evaluate our model results through confusion Matrix.

```r
prediction <- predict(modelFit, newdata=test_part)
confusionMatrix(prediction, test_part$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2037  622  623  557  228
##          B   42  519   44  247  179
##          C  149  377  701  482  398
##          D    0    0    0    0    0
##          E    4    0    0    0  637
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4963          
##                  95% CI : (0.4852, 0.5074)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3418          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9126  0.34190  0.51243   0.0000  0.44175
## Specificity            0.6384  0.91909  0.78296   1.0000  0.99938
## Pos Pred Value         0.5009  0.50339  0.33270      NaN  0.99376
## Neg Pred Value         0.9484  0.85341  0.88378   0.8361  0.88827
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2596  0.06615  0.08934   0.0000  0.08119
## Detection Prevalence   0.5184  0.13140  0.26854   0.0000  0.08170
## Balanced Accuracy      0.7755  0.63049  0.64769   0.5000  0.72056
```
The  algorithm fit a model with accuracy 0.4963. This is a bad and low value. Wi wil tray to fit a new model using Random Forest method. 

## Prediction with Random Forest


```r
modelFit2 <- train(classe ~ ., data = train_part,
                  method = "rf",
                 trControl = fitControl)
print(modelFit2)
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## 
## Summary of sample sizes: 10599, 10598, 10598, 10599, 10597, 10597, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9932634  0.9914776  0.002646153  0.003348517
##   27    0.9966882  0.9958108  0.001673957  0.002117637
##   53    0.9921878  0.9901174  0.003959215  0.005009739
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
prediction2 <- predict(modelFit2, newdata=test_part)
confusionMatrix(prediction2, test_part$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    8    0    0    0
##          B    1 1508    1    0    0
##          C    0    1 1367    6    0
##          D    0    1    0 1280    0
##          E    1    0    0    0 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9976          
##                  95% CI : (0.9962, 0.9985)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9969          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9934   0.9993   0.9953   1.0000
## Specificity            0.9986   0.9997   0.9989   0.9998   0.9998
## Pos Pred Value         0.9964   0.9987   0.9949   0.9992   0.9993
## Neg Pred Value         0.9996   0.9984   0.9998   0.9991   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1922   0.1742   0.1631   0.1838
## Detection Prevalence   0.2852   0.1925   0.1751   0.1633   0.1839
## Balanced Accuracy      0.9988   0.9965   0.9991   0.9976   0.9999
```

The  Random Forest algorithm fit a model with accuracy 0.9976. The out-of-sample error is lower than 0.002 which. It is pretty low.


## Predit with real test set

Finally, we are going to predict the new values in the testing csv provided


```r
prediction3 <- predict(modelFit2, newdata=test)
```


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(prediction3)
```

## Conclusion

We get this prediction appling the model against test set:

```r
print(prediction3)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```









