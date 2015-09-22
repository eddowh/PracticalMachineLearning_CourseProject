# Practical Machine Learning Course Project
Eddo W. Hintoso  
September 22, 2015  




# Processing Data

First and foremost, download the files and load necessary packages:

```r
# load package if not yet loaded
if(!("caret" %in% loadedNamespaces())) {
    library(caret)
}
# download data
if(!file.exists("pml-training.csv")){
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
        destfile = "pml-training.csv")
}
if(!file.exists("pml-testing.csv")){
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
        destfile = "pml-testing.csv")
}
# load data
train <- read.csv('pml-training.csv', header = TRUE, na.strings = c("", "NA", "#DIV/0!"))
test <- read.csv('pml-testing.csv', header = TRUE, na.strings = c("", "NA", "#DIV/0!"))
```

Note that I have treated some string values as `NA`. This is because upon primary inspection of the data there were many missing data that best be uniformly expressed as `NA`. The below R code is shown on how the author came to detect missing values - however it is only for display and will not be evaluated due to the length of the output exceeding the degree of readability. 

```r
# check if there any NA in any column variables of train
TRUE %in% sapply(train, function(col) {NA %in% col})
# individually inspect odd outputs by iterating over factor and character variables
# helps to see the factor levels
for (col in 1:ncol(train)) {
    if(class(train[, col]) == "factor" |
       class(train[, col]) == "character") {
        print(unique(train[, col]))
    }
}
```

In order to run the machine learning algorithms, the features used cannot contain any `NA` values. Let us quickly inspect how many variables are complete:

```r
completionStatus <- sapply(train, function(col) {!(NA %in% col)})
c("Complete Variables" = sum(completionStatus),
  "Total Variables" = length(completionStatus),
  "Percent Completion" = 100 * sum(completionStatus) / length(completionStatus))
```

```
## Complete Variables    Total Variables Percent Completion 
##               60.0              160.0               37.5
```

We now know that only 60 variables have complete data - so we will only use these column variables for our predictor models, since imputing data carries a considerable risk and may affect the accuracy of the predictor model we're going to fit. Think of this as taking a measure to potentially overfit a predictor model due to too many column variables. Now all that is left is filtering the training and testing data sets from the incomplete data columns:

```r
# initialize index to empty vector
complete_index = c()
# iterate from second column since first column is just row index
for (col in 2:length(completionStatus)) {
    if (completionStatus[[col]] == TRUE) {
        complete_index = c(complete_index, col)
    }
}
# filter training and testing dataset into just complete data columns
train <- train[, complete_index]
test <- test[, complete_index]
```

Both the training and testing data should now only have 59 column variables, and this is confirmed below:

```r
c(ncol(train), ncol(test))
```

```
## [1] 59 59
```

However, this isn't the last step to preparing a useful training data set. Upon further inspection, some column variables can be argued to not being contributable to a predicting model.

```r
# infer from variable names
names(train)
```

```
##  [1] "user_name"            "raw_timestamp_part_1" "raw_timestamp_part_2"
##  [4] "cvtd_timestamp"       "new_window"           "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [13] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [16] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [19] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [22] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [25] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [28] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [31] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [34] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [37] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [40] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [43] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [46] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [49] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [52] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [55] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [58] "magnet_forearm_z"     "classe"
```

```r
# examine structure
str(train[, 1:10])
```

```
## 'data.frame':	19622 obs. of  10 variables:
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
```

It can be inferred that the first six variables `user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window` are administrative, integer/factor variables, unlike the other numeric variables that serve to contribute to building a good predictive model. Thus, more variable elimination is required, bringing it down to 53 variables.

```r
# eliminate first 6 columns
train <- train[, -(1:6)]
test <- test[, -(1:6)]
# check dimensions
c(ncol(train), ncol(test))
```

```
## [1] 53 53
```

---

# Cross Validation
