# Load the required libraries
library(caret)                  # Classification and Regression Training 
library(rpart)                  # Recursive Partitioning and Regression Trees
library(rpart.plot)             # Plot 'rpart' Models: An Enhanced Version of 'plot.rpart'
library(randomForest)           # Breiman and Cutler's Random Forests for Classification and Regression
library(rattle)                 # Graphical User Interface for Data Mining in R
library(doMC);                  # Provides a parallel backend for the %dopar% function using

### Download the data

# Create data directory if not exist
if(!file.exists("./data")){dir.create("./data")}

# Download the training data file if it does not exist
if(!file.exists("./data/pml-training.csv")) {
        trainUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(trainUrl, destfile="./data/pml-training.csv", method="curl", quiet=FALSE)
}

# Download the test data file if it does not exist
if(!file.exists("./data/pml-testing.csv")) {
        testUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(testUrl, destfile="./data/pml-testing.csv", method="curl", quiet=FALSE)
}

#Read the .csv files into corresponding data frames.
trainDF <- read.csv("./data/pml-training.csv", na.strings=c("NA","NaN","#DIV/0!", ""))
testDF <- read.csv("./data/pml-testing.csv", na.strings=c("NA","NaN","#DIV/0!", ""))

# Dislay dimentions of the data frames
dim(trainDF)
dim(testDF)
str(trainDF)
str(testDF)

# Remove columns not containing the accelerometer measurements. Keep prediction variable classe.
# Remove columns 1 - 7
reqcol = grep(pattern = "_belt|_arm|_dumbbell|_forearm", names(trainDF))
trainDF <- trainDF[, c(reqcol,160)]
trainDF <- trainDF[, -c(1:7)]
notreqcol = which(colSums(is.na(trainDF)) > 19000)
trainDF = trainDF[, -notreqcol]

# Show table dimensions and show table calsses
dim(trainDF)
table(sapply(trainDF[1,], class))

# Set the seed for reproduceability
set.seed(760924)

inTrain <- createDataPartition(trainDF$classe, p=0.70, list=F)
training <- trainDF[inTrain, ]
testing <- trainDF[-inTrain, ]

# Check the dimentions for each set
dim(training)
dim(testing)

# Fit and train the model
modFitregTree <- rpart(classe ~ ., data=training, minbucket = 2000)

# Plot the tree
fancyRpartPlot(modFitregTree)

# Show predictors
predmodFitregTree <- predict(modFitregTree, testing, type = "class")
confusionMatrix(predmodFitregTree, testing$classe)

#registerDoMC(cores = 4) #Leveraging Multi-Core for Parallelization

# Fit and train the model
modFitrandomForest <- randomForest(classe ~. , data=training, ntree = 500)

# Plot the tree
plot(modFitrandomForest)

# Show predictors
predmodFitrandomForest = predict(modFitrandomForest, newdata = testing)
confusionMatrix(predmodFitrandomForest, testing$classe)

#Dotchart of variable importance as measured by a Random Forest
varImpPlot(modFitrandomForest)

eval <-testDF[,intersect(names(trainDF),names(testDF))] 

predictions<-predict(modFitrandomForest, eval)

pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("answers/problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}







