################################################################################
########################################
### Libraries
########################################
##
## Ensure libraries are downloaded in local R
#
if(!require(tidytext)) 
  install.packages("tidytext", repos = "http://cran.us.r-project.org")
#
if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#
if(!require(readxl)) 
  install.packages("readxl", repos = "http://cran.us.r-project.org")
#
# Ensure Caret for training with dependencies
if(!require(caret)) 
  install.packages("caret", 
                   repos = "http://cran.us.r-project.org",
                   dependencies = TRUE)
#
# httr built for R version 4.4.2 ,
#   but worked for R version 4.4.1
if(!require(httr)) 
  install.packages("httr", repos = "http://cran.us.r-project.org")
#
# Ensure additional libraries for training model are available
# -> These does not need to be loaded using library()
if(!require(naivebayes)) 
  install.packages("naivebayes", repos = "http://cran.us.r-project.org")
#
if(!require(MASS)) 
  install.packages("MASS", repos = "http://cran.us.r-project.org")
#
if(!require(randomForest)) 
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
#
## Load libraries
#
# Most used package in the code 
library(tidyverse)
#
# Package for downloading data from http
library(httr) 
#
# Read csv data files
library(readxl)
#
# Package for text analysis (NLP)
library(tidytext)
#
# Package for training
library(caret)
#
## options
options(digits = 6)
#
#
########################################
### Load Files
########################################
##
# URL of backup zip File in repository
datasetURL <- "https://github.com/IgnisMagnus/FakeNewsAnalysis/raw/refs/heads/main/FakeNewsClassification.zip"
#
## File names
# Zip file containing the dataset
zipFile <- "FakeNewsClassification.zip"
#
# Train, Test, and evaluation data
trainFileName <- "FakeNewsClassification/train (2).csv"
testFileName <- "FakeNewsClassification/test (1).csv"
evalFileName <- "FakeNewsClassification/evaluation.csv"
#
## Make sure data is retrievable (download / unzip)
#
# If zip file does not exist, download from repository backup
if(!file.exists(zipFile)){
  # GET request for the download link
  datasetGET <- GET(datasetURL)
  # Write GET request on a File
  writeBin(content(datasetGET),zipFile)
  # remove var if used.
  rm(datasetGET)
}
#
# Check if files is already available for reading, get it from zip otherwise.
if(!file.exists(trainFileName))
  unzip(zipFile,trainFileName)
#
if(!file.exists(testFileName))
  unzip(zipFile,testFileName)
#
if(!file.exists(evalFileName))
  unzip(zipFile,evalFileName)
#
## Retrieve data set
#
train_set <- read_csv2(trainFileName)
test__set <- read_csv2(testFileName)
eval__set <- read_csv2(evalFileName)
#
## Fix data set
# change '...1' column into index for all three tables.
train_set <- train_set %>% 
  mutate(index = ...1) %>% 
  dplyr::select(index, title, text, label)
test__set <- test__set %>% 
  mutate(index = ...1) %>% 
  dplyr::select(index, title, text, label)
eval__set <- eval__set %>% 
  mutate(index = ...1) %>% 
  dplyr::select(index, title, text, label)
#
## RMs
#
rm(datasetURL)
rm(evalFileName)
rm(testFileName)
rm(trainFileName)
rm(zipFile)
#
#
########################################
### Extract Predictors
########################################
##
#
## Sentiments
#
# afinn : assign positive/negative values to words
senti_a <- get_sentiments("afinn")
#
# nrc   : assign a sentiment category to words
senti_n <- get_sentiments("nrc")
#
## Define functions
#
#
get_title_value <- function(dataset){
  ##
  # Title value : 
  #   Get afinn value for titles
  #   Get sum of afinn values per row
  ##
  dataset %>% 
    # Unnest title to separate words
    unnest_tokens(word,title) %>%
    # Remove stop words (articles, etc.)
    filter(!word %in% stop_words$word) %>% 
    # Select necessary columns
    dplyr::select(index,word) %>% 
    # Join to get assigned 'afinn' value of each word
    left_join(senti_a, by = "word") %>%
    # Words not in afinn are assigned a '0' value
    #   (neither negative nor positive / neutral)
    mutate(value = ifelse(is.na(value), 0, value) ) %>% 
    # Get net value per title
    group_by(index) %>% 
    summarise(title_value = sum(value))
}
get_title_senti <- function(dataset){
  ##
  # Title senti(ments):
  #   Get nrc category per word in title (can be many-to-many)
  #   Get total number of each nrc category
  #   Assign words without nrc category to 'neutral' category
  ##
  dataset %>% 
    # Unnest title to separate words
    unnest_tokens(word,title) %>% 
    # Remove stop words
    filter(!word %in% stop_words$word) %>% 
    # Select necessary columns
    dplyr::select(index,word) %>%
    # Join to get assigned 'nrc' category
    left_join(senti_n, by = "word", relationship = "many-to-many") %>% 
    # Group by index(row) and sentiment then get total number per category
    group_by(index,sentiment) %>% 
    summarise(number = n()) %>% 
    # Assign neutral to words not in 'nrc'
    mutate(sentiment = ifelse(is.na(sentiment), "neutral", sentiment)) %>% 
    # Pivot wider to one observation per row
    pivot_wider(names_from = sentiment,
                values_from = number,
                names_prefix = "title_") %>%
    # Any NAs in categories' columns are set to 0
    replace(is.na(.),0)
}
get_text__value <- function(dataset){
  # Text value : 
  #   Get afinn value for text
  #   Get sum of afinn values per row
  ##
  dataset %>% 
    # Unnest text to separate worsd
    unnest_tokens(word,text) %>% 
    # Remove stop words
    filter(!word %in% stop_words$word) %>% 
    # Select necessary columns
    dplyr::select(index,word) %>% 
    # Join to get assigned 'afinn' value of each word
    left_join(senti_a, by = "word") %>%
    # Words not in afinn are assigned a '0' value
    mutate(value = ifelse(is.na(value), 0, value) ) %>% 
    # Get net value per text/row
    group_by(index) %>% 
    summarise(text_value = sum(value))
}
get_text__senti <- function(dataset){
  ##
  # Text senti(ments):
  #   Get nrc category per word in text (can be many-to-many)
  #   Get total number of each nrc category
  #   Assign words without nrc category to 'neutral' category
  ##
  dataset %>% 
    # Unnest text to separate words
    unnest_tokens(word,text) %>% 
    # Remove stop words
    filter(!word %in% stop_words$word) %>% 
    # Select necessary columns
    dplyr::select(index,word) %>% 
    # Join to get assigned 'nrc' category
    left_join(senti_n, by = "word", relationship = "many-to-many") %>% 
    # Group by index(row) and sentiment then get total number per category
    group_by(index,sentiment) %>% 
    summarise(number = n()) %>%   
    # Assign neutral to words not in 'nrc'
    mutate(sentiment = ifelse(is.na(sentiment), "neutral", sentiment)) %>% 
    # Pivot wider to one observation per row
    pivot_wider(names_from = sentiment,
                values_from = number,
                names_prefix = "text_") %>% 
    # Any NAs in categories' columns are replaced and set to 0
    replace(is.na(.),0)
}
get_title_words <- function(dataset){
  ##
  # Get and Return the number of tokens/words from title
  ##
  dataset %>% 
    # Unnest title for separation
    unnest_tokens(word,title) %>% 
    # Remove Stop words
    filter(!word %in% stop_words$word) %>% 
    # Select index and word only
    dplyr::select(index,word) %>% 
    # count number of words per index/row
    group_by(index) %>% 
    summarise(title_words = n())
}
get_text__words <- function(dataset){
  ##
  # Get and Return the number of token/words from text
  ##
  dataset %>% 
    # Unnest text for separation
    unnest_tokens(word,text) %>%
    # Remove Stop words
    filter(!word %in% stop_words$word) %>% 
    # Select index and word only
    dplyr::select(index,word) %>% 
    # count number of words per index/row
    group_by(index) %>% 
    summarise(text_words = n())
}
get_data_desc <- function(dataset){
  ##
  # Join all predictors' data frames : 
  #   (title_value, title_senti, text__value, 
  #    text__senti, title_words, text__words) into
  #   one data frame with the 'label' column (Fake news or not column)
  # Make sure output column (label) is in factor form for classification train
  ##
  # Get title & text value,senti &words 
  title_value <- get_title_value(dataset)
  title_senti <- get_title_senti(dataset)
  title_words <- get_title_words(dataset)
  text__value <- get_text__value(dataset)
  text__senti <- get_text__senti(dataset)
  text__words <- get_text__words(dataset)
  # Join all predictor data frames
  dataset %>% 
    # Get index and label only
    dplyr::select(index,label) %>% 
    # Using left_join, join each df one by one
    left_join(text__value, by = "index") %>% 
    left_join(text__senti, by = "index") %>% 
    left_join(title_value, by = "index") %>% 
    left_join(title_senti, by = "index") %>% 
    left_join(title_words, by = "index") %>% 
    left_join(text__words, by = "index") %>% 
    # mutate to make (numeric) label into factor
    mutate(label = as.factor(label)) %>% 
    # make sure there is no NAs
    replace(is.na(.),0)
}
#
#
## Form (train) index + label + predictors df
data_description <- get_data_desc(train_set)
#
## Form (test) index + label + predictors df
data_description_t <- get_data_desc(test__set)
#
## Form Eval (evaluation) index + label + predictors df
data_description_E <- get_data_desc(eval__set)
#
## RMs
rm(senti_a,senti_n)
#
#
########################################
### Get important predictors
########################################
##
## Split train & test sets' input and output
# Get input(train) from all predictors
input <- data_description %>% 
  dplyr::select(-label) %>% 
  dplyr::select(-index)
#
# Get a column order
column_order <- colnames(input)[order(colnames(input))]
#
# Reorder columns based on column_order
input <- input %>% 
  dplyr::select(all_of(column_order))
#
# Get output(train) from label column of data_description
output <- data_description$label
#
# Get input_t(test) from predictors
input_t <- data_description_t %>% 
  dplyr::select(-label) %>% 
  dplyr::select(-index)
#
# Reorder input_t columns based on the same column order as input
input_t <- input_t %>% 
  dplyr::select(all_of(column_order))
#
# Get output_t(test) from label column of data_description_t
output_t <- data_description_t$label
#
## Train an rf model, to get important predictors
#
# Train model
set.seed(seed = "1", sample.kind = "Rounding")
rf_model <- caret::train(x = input, y = output, method = "rf")
#
# Get the important vars of the model using varImp
ImportantVar <- varImp(rf_model) 
#
# Get only the column names of the predictors
important_columns <- ImportantVar$importance %>% 
  filter(Overall > 0) %>% 
  rownames()
#
#
########################################
### Form evaluation train set and eval set
########################################
##
## Joined (train + test) so that there is more rows/data for training
data_desc_F <- rbind(data_description,
                     data_description_t %>% 
                       mutate(index = nrow(data_description) + index))
#
## Split large training set into Input and Output
#
# Input : Predictors
input_F <- data_desc_F %>% 
  dplyr::select(-label)
#
# Get only the important_columns
input_F <- input_F %>% 
  dplyr::select(important_columns)
#
# Output : Results
output_F <- data_desc_F$label
#
## Split evaluation set into its own input and output
#
# Get Predictors
input_E <- data_description_E %>% 
  dplyr::select(-label)
#
# Get only the important columns
input_E <- input_E %>% 
  dplyr::select(important_columns)
#
# Get Results ( for evaluating prediction)
output_E <- data_description_E$label
#
#
########################################
### Final Model + Evaluation
########################################
##
## Final RF model
#
set.seed(seed = "1", sample.kind = "Rounding")
ml_final <- caret::train(x = input_F, y = output_F, 
                         method = "rf", tuneGrid = data.frame(mtry = 4))
#
## Predict output
#
predicted_Final <- predict(ml_final,input_E)
#
## Output Statistics
#
# Get confusion Matrix (table + statistics)
cm_Final <- confusionMatrix(data = predicted_Final, reference = output_E, positive = "1")
#
# Get Accuracy
cm_accu <- cm_Final$overall["Accuracy"]
#
# Get other statistics
cm_stat <- cm_Final$byClass[c("Sensitivity", 
                              "Specificity", 
                              "Prevalence")]
#
# Join Accuracy and 3 stats into one data frame
cm_join_t <- cm_accu %>% 
  as.data.frame() %>% 
  rbind(cm_stat %>% as.data.frame())
#
# Retrieve row names (Accuracy,etc..)
cm_names <- cm_join_t %>% rownames()
#
# Transform data frame into horizontal view
final_stats <- data.frame(Stats = cm_names,
                          values = cm_join_t[,1]) %>% 
  mutate(model_name = "Optimized Final RF") %>% 
  pivot_wider(names_from = "Stats",
              values_from = "values")
#
# RMs
rm(cm_join_t,cm_names)
#
#
########################################
### Final Result Table
########################################
## 
## Showing the final statistics of 
##   the evaluation of final model:
#
final_stats %>% knitr::kable()
#
########################################
### End
########################################
################################################################################