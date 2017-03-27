Last week, I introduced how to run machine learning applications on Spark from within R, using the **sparklyr** package. This week, I am showing how to build feed-forward deep neural networks or multilayer perceptrons. The models in this example are built to classify ECG data into being either from *healthy* hearts or from someone suffering from *arrhythmia*. I will show how to prepare a dataset for modeling, setting weights and other modeling parameters and finally, how to evaluate model performance with the **h2o** package via **rsparkling**.

<br>

### Deep learning with neural networks

Deep learning with neural networks is arguably one of the most rapidly growing applications of machine learning and AI today. They allow building complex models that consist of multiple hidden layers within artifical networks and are able to find non-linear patterns in unstructured data. Deep neural networks are usually feed-forward, which means that each layer feeds its output to subsequent layers, but recurrent or feed-back neural networks can also be built. Feed-forward neural networks are also called multilayer perceptrons (MLPs).

<br>

### H2O and Sparkling Water

The R package h2o provides a convient interface to H2O, which is an open-source machine learning and deep learning platform. [H2O](http://www.h2o.ai/h2o/) can be integrated with Apache Spark ([**Sparkling Water**](http://www.h2o.ai/sparkling-water/)) and therefore allows the implementation of complex or big models in a fast and scalable manner. H2O distributes a wide range of common machine learning algorithms for classification, regression and deep learning.

[**Sparkling Water** can be accessed from R](http://spark.rstudio.com/h2o.html) with the **rsparkling** extension package to **sparklyr** and **h2o**. Check the documentation for **rsparkling** to find out which H2O, Sparkling Water and Spark versions are compatible.

<br>

### Preparing the R session

First, we need to load the packages and connect to the Spark instance (for demontration purposes, I am using a local instance).

``` r
library(rsparkling)
options(rsparkling.sparklingwater.version = "2.0.3")

library(h2o)
library(dplyr)
library(sparklyr)

sc <- spark_connect(master = "local", version = "2.0.0")
```

I am also preparing my custom plotting theme.

``` r
library(ggplot2)
library(ggrepel)

my_theme <- function(base_size = 12, base_family = "sans"){
  theme_minimal(base_size = base_size, base_family = base_family) +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    panel.grid.major = element_line(color = "grey"),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "aliceblue"),
    strip.background = element_rect(fill = "darkgrey", color = "grey", size = 1),
    strip.text = element_text(face = "bold", size = 12, color = "white"),
    legend.position = "right",
    legend.justification = "top", 
    panel.border = element_rect(color = "grey", fill = NA, size = 0.5)
  )
}
```

<br>

### Arrhythmia data

The data I am using to demonstrate the building of neural nets is the arrhythmia dataset from [UC Irvine's machine learning database](https://archive.ics.uci.edu/ml/datasets/Arrhythmia). It contains 279 features from ECG heart rhythm diagnostics and one output column. I am not going to rename the feature columns because they are too many and the descriptions are too complex. Also, we don't need to know specifically which features we are looking at for building the models. For a description of each feature, see <https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.names>. The output column defines 16 classes: class 1 samples are from healthy ECGs, the remaining classes belong to different types of arrhythmia, with class 16 being all remaining arrhythmia cases that didn't fit into distinct classes.

``` r
arrhythmia <- read.table("arrhythmia.data.txt", sep = ",")

# making sure, that all feature columns are numeric
arrhythmia[-280] <- lapply(arrhythmia[-280], as.numeric)

#  renaming output column and converting to factor
colnames(arrhythmia)[280] <- "class"
arrhythmia$class <- as.factor(arrhythmia$class)
```

As usual, I want to get acquainted with the data and explore it's properties before I am building any model. So, I am first going to look at the distribution of classes and of healthy and arrhythmia samples.

``` r
p1 <- ggplot(arrhythmia, aes(x = class)) +
  geom_bar(fill = "navy", alpha = 0.7) +
  my_theme()
```

Because I am interested in distinguishing healthy from arrhythmia ECGs, I am converting the output to binary format by combining all arrhythmia cases into one class.

``` r
arrhythmia$diagnosis <- ifelse(arrhythmia$class == 1, "healthy", "arrhythmia")
arrhythmia$diagnosis <- as.factor(arrhythmia$diagnosis)
```

``` r
p2 <- ggplot(arrhythmia, aes(x = diagnosis)) +
  geom_bar(fill = "navy", alpha = 0.7) +
  my_theme()
```

``` r
library(gridExtra)
library(grid)

grid.arrange(p1, p2, ncol = 2)
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-8-1.png" style="display: block; margin: auto;" />

With binary classification, we have almost the same numbers of healthy and arrhythmia cases in our dataset.

I am also interested in how much the normal and arrhythmia cases cluster in a Principal Component Analysis (PCA). I am first preparing the PCA plotting function and then run it on the feature data.

``` r
library(pcaGoPromoter)

pca_func <- function(pcaOutput2, group_name){
    centroids <- aggregate(cbind(PC1, PC2) ~ groups, pcaOutput2, mean)
    conf.rgn  <- do.call(rbind, lapply(unique(pcaOutput2$groups), function(t)
          data.frame(groups = as.character(t),
                     ellipse(cov(pcaOutput2[pcaOutput2$groups == t, 1:2]),
                           centre = as.matrix(centroids[centroids$groups == t, 2:3]),
                           level = 0.95),
                     stringsAsFactors = FALSE)))
        
    plot <- ggplot(data = pcaOutput2, aes(x = PC1, y = PC2, group = groups, color = groups)) + 
      geom_polygon(data = conf.rgn, aes(fill = groups), alpha = 0.2) +
      geom_point(size = 2, alpha = 0.5) + 
      labs(color = paste(group_name),
           fill = paste(group_name),
           x = paste0("PC1: ", round(pcaOutput$pov[1], digits = 2) * 100, "% variance"),
           y = paste0("PC2: ", round(pcaOutput$pov[2], digits = 2) * 100, "% variance")) +
      my_theme()
    
    return(plot)
}
```

``` r
pcaOutput <- pca(t(arrhythmia[-c(280, 281)]), printDropped = FALSE, scale = TRUE, center = TRUE)
pcaOutput2 <- as.data.frame(pcaOutput$scores)

pcaOutput2$groups <- arrhythmia$class
p1 <- pca_func(pcaOutput2, group_name = "class")

pcaOutput2$groups <- arrhythmia$diagnosis
p2 <- pca_func(pcaOutput2, group_name = "diagnosis")

grid.arrange(p1, p2, ncol = 2)
```

![](h2o_files/figure-markdown_github/unnamed-chunk-10-1.png)

The PCA shows that there is a big overlap between healthy and arrhythmia samples, i.e. there does not seem to be major global differences in all features. The class that is most distinct from all others seems to be class 9. I want to give the arrhythmia cases that are very different from the rest a stronger weight in the neural network, so I define a weight column where every sample outside the central PCA cluster will get a "2", they will in effect be used twice in the model.

``` r
weights <- ifelse(pcaOutput2$PC1 < -5 & abs(pcaOutput2$PC2) > 10, 2, 1)
```

I also want to know what the variance is within features.

``` r
library(matrixStats)

colvars <- data.frame(feature = colnames(arrhythmia[-c(280, 281)]),
                      variance = colVars(as.matrix(arrhythmia[-c(280, 281)])))

subset(colvars, variance > 50) %>%
  mutate(feature = factor(feature, levels = colnames(arrhythmia[-c(280, 281)]))) %>%
  ggplot(aes(x = feature, y = variance)) +
    geom_bar(stat = "identity", fill = "navy", alpha = 0.7) +
    my_theme() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-12-1.png" style="display: block; margin: auto;" />

Features with low variance are less likely to strongly contribute to a differentiation between healthy and arrhthmia cases, so I am going to remove them. I am also concatenating the weights column:

``` r
arrhythmia_subset <- cbind(weights, arrhythmia[, c(281, 280, which(colvars$variance > 50))])
```

<br>

### Working with rsparkling and h2o

Now that I have my final dataframe for modeling, I copy it to the Spark instance. For working with **h2o** functions, the data needs to be converted from a Spark DataFrame to an H2O Frame. This is done with the *as\_h2o\_frame()* function.

``` r
arrhythmia_sc <- copy_to(sc, arrhythmia_subset)
arrhythmia_hf <- as_h2o_frame(sc, arrhythmia_sc, strict_version_check = FALSE)
```

We can now access all functions from the **h2o** package that are built to work on H2O Frames. A useful such function is *h2o.describe()*. It is similar to base R's *summary()* function but outputs many more descriptive measures for our data. To get a good overview about these measures, I am going to plot them.

``` r
library(tidyr) # for gathering
h2o.describe(arrhythmia_hf[, -1]) %>% # excluding the weights column
  gather(x, y, Zeros:Sigma) %>%
  mutate(group = ifelse(x %in% c("Min", "Max", "Mean"), "min, mean, max", 
                        ifelse(x %in% c("NegInf", "PosInf"), "Inf", "sigma, zeros"))) %>% # separating them into facets makes them easier to see
  mutate(Label = factor(Label, levels = colnames(arrhythmia_hf[, -1]))) %>%
  ggplot(aes(x = Label, y = as.numeric(y), color = x)) +
    geom_point(size = 4, alpha = 0.6) +
    scale_color_brewer(palette = "Set1") +
    my_theme() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    facet_grid(group ~ ., scales = "free") +
    labs(x = "Feature",
         y = "Value",
         color = "")
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-16-1.png" style="display: block; margin: auto;" />

I am also interested in the correlation between features and the output. We can use the *h2o.cor()* function to calculate the correlation matrix. It is again much easier to understand the data when we visualize it, so I am going to create another plot.

``` r
library(reshape2) # for melting

arrhythmia_hf[, 2] <- h2o.asfactor(arrhythmia_hf[, 2]) # diagnosis is now a characer column and we need to convert it again
arrhythmia_hf[, 3] <- h2o.asfactor(arrhythmia_hf[, 3]) # same for class

cor <- h2o.cor(arrhythmia_hf[, -c(1, 3)])
rownames(cor) <- colnames(cor)

melt(cor) %>%
  mutate(Var2 = rep(rownames(cor), nrow(cor))) %>%
  mutate(Var2 = factor(Var2, levels = colnames(cor))) %>%
  mutate(variable = factor(variable, levels = colnames(cor))) %>%
  ggplot(aes(x = variable, y = Var2, fill = value)) + 
    geom_tile(width = 0.9, height = 0.9) +
    scale_fill_gradient2(low = "white", high = "red", name = "Cor.") +
    my_theme() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    labs(x = "", 
         y = "")
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-17-1.png" style="display: block; margin: auto;" />

<br>

### Training, test and validation data

Now we can use the *h2o.splitFrame()* function to split the data into training, validation and test data. Here, I am using 70% for training and 15% each for validation and testing. We could also just split the data into two sections, a training and test set but when we have sufficient samples, it is a good idea to evaluate model performance on an independent test set on top of training with a validation set. Because we can easily overfit a model, we want to get an idea about how generalizable it is - this we can only assess by looking at how well it works on previously unknown data.

I am also defining reponse, feature and weight column names now.

``` r
splits <- h2o.splitFrame(arrhythmia_hf, 
                         ratios = c(0.7, 0.15), 
                         seed = 1)

train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

response <- "diagnosis"
weights <- "weights"
features <- setdiff(colnames(train), c(response, weights, "class"))
```

``` r
summary(train$diagnosis, exact_quantiles = TRUE)
```

    ##  diagnosis      
    ##  healthy   :163 
    ##  arrhythmia:155

``` r
summary(valid$diagnosis, exact_quantiles = TRUE)
```

    ##  diagnosis     
    ##  healthy   :43 
    ##  arrhythmia:25

``` r
summary(test$diagnosis, exact_quantiles = TRUE)
```

    ##  diagnosis     
    ##  healthy   :39 
    ##  arrhythmia:27

If we had more categorical features, we could use the *h2o.interaction()* function to define interaction terms, but since we only have numeric features here, we don't need this.

We can also run a PCA on the training data, using the *h2o.prcomp()* function to calculate the singular value decomposition of the Gram matrix with the power method.

``` r
pca <- h2o.prcomp(training_frame = train,
           x = features,
           validation_frame = valid,
           transform = "NORMALIZE",
           k = 3,
           seed = 42)
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=============                                                    |  20%
      |                                                                       
      |=================================================================| 100%

``` r
pca
```

    ## Model Details:
    ## ==============
    ## 
    ## H2ODimReductionModel: pca
    ## Model ID:  PCA_model_R_1488113493291_1 
    ## Importance of components: 
    ##                             pc1      pc2      pc3
    ## Standard deviation     0.598791 0.516364 0.424850
    ## Proportion of Variance 0.162284 0.120680 0.081695
    ## Cumulative Proportion  0.162284 0.282965 0.364660
    ## 
    ## 
    ## H2ODimReductionMetrics: pca
    ## 
    ## No model metrics available for PCA
    ## H2ODimReductionMetrics: pca
    ## 
    ## No model metrics available for PCA

``` r
eigenvec <- as.data.frame(pca@model$eigenvectors)
eigenvec$label <- features

ggplot(eigenvec, aes(x = pc1, y = pc2, label = label)) +
  geom_point(color = "navy", alpha = 0.7) +
  geom_text_repel() +
  my_theme()
```

![](h2o_files/figure-markdown_github/unnamed-chunk-23-1.png)

<br>

### Modeling

Now, we can build a deep neural network model. We can specifiy quite a few parameters, like

-   **Cross-validation**: Cross validation can tell us the training and validation errors for each model. The final model will be overwritten with the best model, if we don't specify otherwise.

-   **Adaptive learning rate**: For deep learning with h2o, we by default use stochastic gradient descent optimization with an an adaptive learning rate. The two corresponding parameters *rho* and *epsilon* help us find global (or near enough) optima.

-   **Activation function**: The activation function defines the node output relative to a given set of inputs. We want our activation function to be non-linear and continuously differentiable.

-   **Hidden nodes**: Defines the number of hidden layers and the number of nodes per layer.

-   **Epochs**: Increasing the number of epochs (one full training cycle on all training samples) can increase model performance, but we also run the risk of overfitting. To determine the optimal number of epochs, we need to use early stopping.

-   **Early stopping**: By default, early stopping is enabled. This means that training will be stopped when we reach a certain validation error to prevent overfitting.

Of course, you need quite a bit of experience and intuition to hit on a good combination of parameters. That's why it usually makes sense to do a grid search for hyper-parameter tuning. Here, I want to focus on building and evaluating deep learning models, though. I will cover grid search in next week's post.

``` r
dl_model <- h2o.deeplearning(x = features,
                             y = response,
                             weights_column = weights,
                             model_id = "dl_model",
                             training_frame = train,
                             validation_frame = valid,
                             nfolds = 15,                                   # 10x cross validation
                             keep_cross_validation_fold_assignment = TRUE,
                             fold_assignment = "Stratified",
                             activation = "RectifierWithDropout",
                             score_each_iteration = TRUE,
                             hidden = c(200, 200, 200, 200, 200),           # 5 hidden layers, each of 200 neurons
                             epochs = 100,
                             variable_importances = TRUE,
                             export_weights_and_biases = TRUE,
                             seed = 42)
```

Because training can take a while, depending on how many samples, features, nodes and hidden layers you are training on, it is a good idea to save your model.

``` r
h2o.saveModel(dl_model, path="dl_model", force = TRUE)
```

We can then re-load the model again any time to check the model quality and make predictions on new data.

``` r
dl_model <- h2o.loadModel("/Users/Shirin/Documents/Github/blog_posts_prep/wip/nn/dl_model/dl_model")
```

<br>

### Model performance

We now want to know how our model performed on the validation data. The *summary()* function will give us a detailed overview of our model.

``` r
summary(dl_model)
```

    ## Model Details:
    ## ==============
    ## 
    ## H2OBinomialModel: deeplearning
    ## Model Key:  dl_model 
    ## Status of Neuron Layers: predicting diagnosis, 2-class classification, bernoulli distribution, CrossEntropy loss, 179,402 weights/biases, 2.1 MB, 33,790 training samples, mini-batch size 1
    ##   layer units             type dropout       l1       l2 mean_rate
    ## 1     1    90            Input  0.00 %                            
    ## 2     2   200 RectifierDropout 50.00 % 0.000000 0.000000  0.007454
    ## 3     3   200 RectifierDropout 50.00 % 0.000000 0.000000  0.007665
    ## 4     4   200 RectifierDropout 50.00 % 0.000000 0.000000  0.009283
    ## 5     5   200 RectifierDropout 50.00 % 0.000000 0.000000  0.008444
    ## 6     6   200 RectifierDropout 50.00 % 0.000000 0.000000  0.018733
    ## 7     7     2          Softmax         0.000000 0.000000  0.002265
    ##   rate_rms momentum mean_weight weight_rms mean_bias bias_rms
    ## 1                                                            
    ## 2 0.007066 0.000000    0.002824   0.095555  0.427771 0.052424
    ## 3 0.003919 0.000000   -0.007436   0.074584  0.952129 0.050540
    ## 4 0.004152 0.000000   -0.007882   0.071979  0.965512 0.028140
    ## 5 0.004196 0.000000   -0.006135   0.070836  0.973472 0.029246
    ## 6 0.040437 0.000000   -0.010248   0.070627  0.953217 0.032597
    ## 7 0.000968 0.000000   -0.041048   0.377904 -0.000315 0.065418
    ## 
    ## H2OBinomialMetrics: deeplearning
    ## ** Reported on training data. **
    ## ** Metrics reported on full training frame **
    ## 
    ## MSE:  0.03011144
    ## RMSE:  0.1735265
    ## LogLoss:  0.1147594
    ## Mean Per-Class Error:  0.0304878
    ## AUC:  0.9864582
    ## Gini:  0.9729164
    ## 
    ## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
    ##            arrhythmia healthy    Error     Rate
    ## arrhythmia        154      10 0.060976  =10/164
    ## healthy             0     163 0.000000   =0/163
    ## Totals            154     173 0.030581  =10/327
    ## 
    ## Maximum Metrics: Maximum metrics at their respective thresholds
    ##                         metric threshold    value idx
    ## 1                       max f1  0.624219 0.970238 172
    ## 2                       max f2  0.624219 0.987879 172
    ## 3                 max f0point5  0.940109 0.969962 158
    ## 4                 max accuracy  0.868278 0.969419 168
    ## 5                max precision  0.984610 1.000000   0
    ## 6                   max recall  0.624219 1.000000 172
    ## 7              max specificity  0.984610 1.000000   0
    ## 8             max absolute_mcc  0.624219 0.940609 172
    ## 9   max min_per_class_accuracy  0.922562 0.963190 162
    ## 10 max mean_per_class_accuracy  0.624219 0.969512 172
    ## 
    ## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
    ## H2OBinomialMetrics: deeplearning
    ## ** Reported on validation data. **
    ## ** Metrics reported on full validation frame **
    ## 
    ## MSE:  0.1687593
    ## RMSE:  0.4108032
    ## LogLoss:  0.8137165
    ## Mean Per-Class Error:  0.2232558
    ## AUC:  0.852093
    ## Gini:  0.704186
    ## 
    ## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
    ##            arrhythmia healthy    Error    Rate
    ## arrhythmia         15      10 0.400000  =10/25
    ## healthy             2      41 0.046512   =2/43
    ## Totals             17      51 0.176471  =12/68
    ## 
    ## Maximum Metrics: Maximum metrics at their respective thresholds
    ##                         metric threshold    value idx
    ## 1                       max f1  0.004259 0.872340  50
    ## 2                       max f2  0.000026 0.942982  55
    ## 3                 max f0point5  0.558018 0.844156  46
    ## 4                 max accuracy  0.558018 0.823529  46
    ## 5                max precision  0.984052 1.000000   0
    ## 6                   max recall  0.000026 1.000000  55
    ## 7              max specificity  0.984052 1.000000   0
    ## 8             max absolute_mcc  0.004259 0.616316  50
    ## 9   max min_per_class_accuracy  0.949218 0.720000  37
    ## 10 max mean_per_class_accuracy  0.558018 0.793488  46
    ## 
    ## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
    ## H2OBinomialMetrics: deeplearning
    ## ** Reported on cross-validation data. **
    ## ** 15-fold cross-validation on training data (Metrics computed for combined holdout predictions) **
    ## 
    ## MSE:  0.162591
    ## RMSE:  0.4032257
    ## LogLoss:  0.55433
    ## Mean Per-Class Error:  0.2257781
    ## AUC:  0.8577735
    ## Gini:  0.7155469
    ## 
    ## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
    ##            arrhythmia healthy    Error     Rate
    ## arrhythmia         99      65 0.396341  =65/164
    ## healthy             9     154 0.055215   =9/163
    ## Totals            108     219 0.226300  =74/327
    ## 
    ## Maximum Metrics: Maximum metrics at their respective thresholds
    ##                         metric threshold    value idx
    ## 1                       max f1  0.148967 0.806283 218
    ## 2                       max f2  0.140425 0.886728 221
    ## 3                 max f0point5  0.719293 0.801825 150
    ## 4                 max accuracy  0.719293 0.792049 150
    ## 5                max precision  0.945732 0.894737  75
    ## 6                   max recall  0.000082 1.000000 301
    ## 7              max specificity  0.998558 0.993902   0
    ## 8             max absolute_mcc  0.719293 0.585581 150
    ## 9   max min_per_class_accuracy  0.647902 0.779141 162
    ## 10 max mean_per_class_accuracy  0.719293 0.791935 150
    ## 
    ## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
    ## Cross-Validation Metrics Summary: 
    ##                               mean          sd  cv_1_valid  cv_2_valid
    ## accuracy                0.86104757  0.05827685   0.6785714   0.7647059
    ## auc                     0.87845755   0.0651871  0.72727275   0.6666667
    ## err                     0.13895245  0.05827685  0.32142857  0.23529412
    ## err_count                3.2666667    1.702286         9.0         4.0
    ## f0point5                0.85721153  0.08901711   0.6043956   0.6060606
    ## f1                      0.86378616  0.06226583   0.7096774   0.6666667
    ## f2                      0.87850237 0.048980236    0.859375   0.7407407
    ## lift_top_group           1.7281693  0.54166335   2.5454545         0.0
    ## logloss                 0.54315174  0.21299319  0.88122135   1.3078347
    ## max_per_class_error     0.23033981 0.084703796   0.5294118        0.25
    ## mcc                     0.73816645  0.10380194  0.50874704  0.50920105
    ## mean_per_class_accuracy  0.8643971 0.052563712   0.7352941       0.775
    ## mean_per_class_error     0.1356029 0.052563712   0.2647059       0.225
    ## mse                     0.15726262  0.04539193   0.2989567  0.26755697
    ## precision                0.8566567  0.10882052        0.55   0.5714286
    ## r2                       0.3370781  0.20342791 -0.25337994 -0.28873268
    ## recall                    0.894164 0.059801217         1.0         0.8
    ## rmse                    0.38896403 0.054633364   0.5467693  0.51725906
    ## specificity             0.83463013  0.11812946  0.47058824        0.75
    ##                         cv_3_valid  cv_4_valid cv_5_valid  cv_6_valid
    ## accuracy                 0.8666667   0.9130435  0.8888889      0.9375
    ## auc                      0.8888889  0.95238096  0.9220779    0.953125
    ## err                     0.13333334  0.08695652 0.11111111      0.0625
    ## err_count                      2.0         2.0        2.0         1.0
    ## f0point5                 0.9459459   0.9677419  0.9574468  0.90909094
    ## f1                           0.875   0.9230769        0.9   0.9411765
    ## f2                      0.81395346  0.88235295  0.8490566   0.9756098
    ## lift_top_group           1.6666666   1.6428572  1.6363636         2.0
    ## logloss                 0.42629975  0.36616853 0.36161652  0.24766968
    ## max_per_class_error     0.22222222  0.14285715 0.18181819       0.125
    ## mcc                      0.7637626   0.8374358   0.797724   0.8819171
    ## mean_per_class_accuracy  0.8888889   0.9285714 0.90909094      0.9375
    ## mean_per_class_error    0.11111111 0.071428575 0.09090909      0.0625
    ## mse                      0.1318371  0.11595513 0.12883213 0.073407516
    ## precision                      1.0         1.0        1.0   0.8888889
    ## r2                      0.45067874   0.5131725 0.45790118  0.70636994
    ## recall                   0.7777778  0.85714287  0.8181818         1.0
    ## rmse                     0.3630938  0.34052184 0.35893193  0.27093822
    ## specificity                    1.0         1.0        1.0       0.875
    ##                          cv_7_valid cv_8_valid cv_9_valid cv_10_valid
    ## accuracy                 0.95454544 0.90909094 0.84615386   0.8333333
    ## auc                      0.97321427  0.9285714  0.8496732       0.875
    ## err                     0.045454547 0.09090909 0.15384616  0.16666667
    ## err_count                       1.0        1.0        4.0         5.0
    ## f0point5                  0.9722222  0.8974359 0.86021507   0.7777778
    ## f1                       0.93333334 0.93333334  0.8888889   0.8484849
    ## f2                        0.8974359  0.9722222  0.9195402  0.93333334
    ## lift_top_group                 2.75  1.5714285  1.5294118    2.142857
    ## logloss                  0.31371167 0.36771947 0.54875386  0.51765615
    ## max_per_class_error           0.125       0.25 0.33333334      0.3125
    ## mcc                       0.9036961 0.81009257  0.6519457   0.7117436
    ## mean_per_class_accuracy      0.9375      0.875  0.8039216     0.84375
    ## mean_per_class_error         0.0625      0.125 0.19607843     0.15625
    ## mse                      0.10615891 0.13233028 0.15460968  0.15598302
    ## precision                       1.0      0.875 0.84210527   0.7368421
    ## r2                        0.5412418 0.42814416 0.31688797  0.37328252
    ## recall                        0.875        1.0  0.9411765         1.0
    ## rmse                      0.3258204 0.36377227 0.39320436  0.39494684
    ## specificity                     1.0       0.75  0.6666667      0.6875
    ##                         cv_11_valid  cv_12_valid cv_13_valid cv_14_valid
    ## accuracy                 0.90909094    0.7619048     0.78125  0.87096775
    ## auc                       0.9338843   0.75454545   0.8515625         0.9
    ## err                      0.09090909   0.23809524     0.21875  0.12903225
    ## err_count                       2.0          5.0         7.0         4.0
    ## f0point5                  0.9574468    0.7258065  0.77380955   0.9027778
    ## f1                              0.9    0.7826087   0.7878788   0.8666667
    ## f2                        0.8490566    0.8490566  0.80246913   0.8333333
    ## lift_top_group                  2.0          0.0         2.0      1.9375
    ## logloss                   0.3949628    1.1107926   0.5032864   0.5132355
    ## max_per_class_error      0.18181819   0.36363637        0.25      0.1875
    ## mcc                       0.8320503    0.5516187  0.56360185  0.74896055
    ## mean_per_class_accuracy  0.90909094    0.7681818     0.78125  0.87291664
    ## mean_per_class_error     0.09090909   0.23181818     0.21875  0.12708333
    ## mse                      0.14628646   0.25551543  0.16808373  0.14408164
    ## precision                       1.0    0.6923077   0.7647059   0.9285714
    ## r2                       0.41485414 -0.024384623   0.3276651  0.42307308
    ## recall                    0.8181818          0.9      0.8125      0.8125
    ## rmse                     0.38247412   0.50548536  0.40998015   0.3795809
    ## specificity                     1.0    0.6363636        0.75  0.93333334
    ##                         cv_15_valid
    ## accuracy                        1.0
    ## auc                             1.0
    ## err                             0.0
    ## err_count                       0.0
    ## f0point5                        1.0
    ## f1                              1.0
    ## f2                              1.0
    ## lift_top_group                  2.5
    ## logloss                  0.28634688
    ## max_per_class_error             0.0
    ## mcc                             1.0
    ## mean_per_class_accuracy         1.0
    ## mean_per_class_error            0.0
    ## mse                     0.079344615
    ## precision                       1.0
    ## r2                        0.6693974
    ## recall                          1.0
    ## rmse                     0.28168175
    ## specificity                     1.0
    ## 
    ## Scoring History: 
    ##              timestamp          duration training_speed    epochs
    ## 1  2017-02-26 12:00:09         0.000 sec                  0.00000
    ## 2  2017-02-26 12:00:13  9 min 19.276 sec    960 obs/sec  10.62579
    ## 3  2017-02-26 12:00:17  9 min 23.224 sec    922 obs/sec  21.25157
    ## 4  2017-02-26 12:00:20  9 min 26.855 sec    936 obs/sec  31.87736
    ## 5  2017-02-26 12:00:23  9 min 29.914 sec    983 obs/sec  42.50314
    ## 6  2017-02-26 12:00:26  9 min 32.962 sec   1016 obs/sec  53.12893
    ## 7  2017-02-26 12:00:30  9 min 36.215 sec   1027 obs/sec  63.75472
    ## 8  2017-02-26 12:00:33  9 min 39.080 sec   1053 obs/sec  74.38050
    ## 9  2017-02-26 12:00:36  9 min 42.416 sec   1054 obs/sec  85.00629
    ## 10 2017-02-26 12:00:39  9 min 45.618 sec   1060 obs/sec  95.63208
    ## 11 2017-02-26 12:00:43  9 min 49.065 sec   1058 obs/sec 106.25786
    ##    iterations      samples training_rmse training_logloss training_auc
    ## 1           0     0.000000                                            
    ## 2           1  3379.000000       0.37492          0.52964      0.89503
    ## 3           2  6758.000000       0.33088          0.41127      0.92780
    ## 4           3 10137.000000       0.28624          0.27850      0.94879
    ## 5           4 13516.000000       0.29155          0.31936      0.95331
    ## 6           5 16895.000000       0.26624          0.24892      0.96319
    ## 7           6 20274.000000       0.25484          0.22395      0.96966
    ## 8           7 23653.000000       0.21655          0.16974      0.97236
    ## 9           8 27032.000000       0.20615          0.16014      0.97737
    ## 10          9 30411.000000       0.19450          0.13990      0.98130
    ## 11         10 33790.000000       0.17353          0.11476      0.98646
    ##    training_lift training_classification_error validation_rmse
    ## 1                                                             
    ## 2        1.50460                       0.17125         0.44104
    ## 3        2.00613                       0.13150         0.42684
    ## 4        1.50460                       0.09786         0.39884
    ## 5        2.00613                       0.08869         0.41348
    ## 6        2.00613                       0.07339         0.40066
    ## 7        2.00613                       0.06422         0.42174
    ## 8        2.00613                       0.05199         0.43371
    ## 9        2.00613                       0.04281         0.42489
    ## 10       2.00613                       0.03364         0.42561
    ## 11       2.00613                       0.03058         0.41080
    ##    validation_logloss validation_auc validation_lift
    ## 1                                                   
    ## 2             0.90435        0.80837         1.58140
    ## 3             0.80734        0.82884         1.58140
    ## 4             0.58092        0.89953         1.58140
    ## 5             0.74467        0.86977         1.58140
    ## 6             0.67012        0.86047         1.58140
    ## 7             0.75170        0.86140         1.58140
    ## 8             0.84701        0.85395         1.58140
    ## 9             0.84022        0.84372         1.58140
    ## 10            0.85505        0.85302         1.58140
    ## 11            0.81372        0.85209         1.58140
    ##    validation_classification_error
    ## 1                                 
    ## 2                          0.20588
    ## 3                          0.19118
    ## 4                          0.17647
    ## 5                          0.17647
    ## 6                          0.17647
    ## 7                          0.17647
    ## 8                          0.16176
    ## 9                          0.17647
    ## 10                         0.17647
    ## 11                         0.17647
    ## 
    ## Variable Importances: (Extract with `h2o.varimp`) 
    ## =================================================
    ## 
    ## Variable Importances: 
    ##   variable relative_importance scaled_importance percentage
    ## 1     V169            1.000000          1.000000   0.013925
    ## 2     V239            0.987290          0.987290   0.013748
    ## 3     V103            0.913953          0.913953   0.012727
    ## 4      V15            0.907422          0.907422   0.012636
    ## 5      V91            0.904267          0.904267   0.012592
    ## 
    ## ---
    ##    variable relative_importance scaled_importance percentage
    ## 85      V88            0.717914          0.717914   0.009997
    ## 86     V269            0.715800          0.715800   0.009968
    ## 87     V137            0.712923          0.712923   0.009928
    ## 88     V168            0.711402          0.711402   0.009906
    ## 89      V33            0.707356          0.707356   0.009850
    ## 90     V219            0.696149          0.696149   0.009694

One performance metric we are interested in is the mean per class error for training and validation data.

``` r
h2o.mean_per_class_error(dl_model, train = TRUE, valid = TRUE, xval = TRUE)
```

    ##     train     valid      xval 
    ## 0.0304878 0.2232558 0.2257781

The confusion matrix tells us, how many classes have been predicted correctly and how many predictions were accurate. Here, we see the errors in predictions on validation data

``` r
h2o.confusionMatrix(dl_model, valid = TRUE)
```

    ## Confusion Matrix (vertical: actual; across: predicted)  for max f1 @ threshold = 0.00425904880659062:
    ##            arrhythmia healthy    Error    Rate
    ## arrhythmia         15      10 0.400000  =10/25
    ## healthy             2      41 0.046512   =2/43
    ## Totals             17      51 0.176471  =12/68

We can also plot the classification error over all epochs or samples.

``` r
plot(dl_model,
     timestep = "epochs",
     metric = "classification_error")
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-33-1.png" style="display: block; margin: auto;" />

``` r
plot(dl_model,
     timestep = "samples",
     metric = "classification_error")
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-34-1.png" style="display: block; margin: auto;" />

Next to the classification error, we are usually interested in the logistic loss (negative log-likelihood or log loss). It describes the sum of errors for each sample in the training or validation data or the negative logarithm of the likelihood of error for a given prediction/ classification. Simply put, the lower the loss, the better the model (if we ignore potential overfitting).

``` r
plot(dl_model,
     timestep = "epochs",
     metric = "logloss")
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-36-1.png" style="display: block; margin: auto;" />

We can also plot the mean squared error (MSE). The MSE tells us the average of the prediction errors squared, i.e. the estimator's variance and bias. The closer to zero, the better a model.

``` r
plot(dl_model,
     timestep = "epochs",
     metric = "rmse")
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-37-1.png" style="display: block; margin: auto;" />

Next, we want to know the area under the curve (AUC). AUC is an important metric for measuring binary classification model performances. It gives the area under the curve, i.e. the integral, of true positive vs false positive rates. The closer to 1, the better a model. AUC is especially useful, when we have unbalanced datasets (meaning datasets where one class is much more common than the other), because it is independent of class labels.

``` r
h2o.auc(dl_model, train = TRUE)
```

    ## [1] 0.9864582

``` r
h2o.auc(dl_model, valid = TRUE)
```

    ## [1] 0.852093

``` r
h2o.auc(dl_model, xval = TRUE)
```

    ## [1] 0.8577735

The weights for connecting two adjacent layers and per-neuron biases that we specified the model to save, can be accessed with:

``` r
w <- h2o.weights(dl_model, matrix_id = 1)
b <- h2o.biases(dl_model, vector_id = 1)
```

Variable importance can be extracted as well (but keep in mind, that variable importance in deep neural networks is difficult to assess and should be considered only as rough estimates).

``` r
h2o.varimp(dl_model)
```

    ## Variable Importances: 
    ##   variable relative_importance scaled_importance percentage
    ## 1     V169            1.000000          1.000000   0.013925
    ## 2     V239            0.987290          0.987290   0.013748
    ## 3     V103            0.913953          0.913953   0.012727
    ## 4      V15            0.907422          0.907422   0.012636
    ## 5      V91            0.904267          0.904267   0.012592
    ## 
    ## ---
    ##    variable relative_importance scaled_importance percentage
    ## 85      V88            0.717914          0.717914   0.009997
    ## 86     V269            0.715800          0.715800   0.009968
    ## 87     V137            0.712923          0.712923   0.009928
    ## 88     V168            0.711402          0.711402   0.009906
    ## 89      V33            0.707356          0.707356   0.009850
    ## 90     V219            0.696149          0.696149   0.009694

``` r
#h2o.varimp_plot(dl_model)
```

<br>

#### Test data

Now that we have a good idea about model performance on validation data, we want to know how it performed on unseen test data. A good model should find an optimal balance between accuracy on training and test data. A model that has 0% error on the training data but 40% error on the test data is in effect useless. It overfit on the training data and is thus not able to generalize to unknown data.

``` r
perf <- h2o.performance(dl_model, test)
perf
```

    ## H2OBinomialMetrics: deeplearning
    ## 
    ## MSE:  0.2154912
    ## RMSE:  0.4642103
    ## LogLoss:  1.378809
    ## Mean Per-Class Error:  0.2250712
    ## AUC:  0.7796771
    ## Gini:  0.5593542
    ## 
    ## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
    ##            arrhythmia healthy    Error    Rate
    ## arrhythmia         19       8 0.296296   =8/27
    ## healthy             6      33 0.153846   =6/39
    ## Totals             25      41 0.212121  =14/66
    ## 
    ## Maximum Metrics: Maximum metrics at their respective thresholds
    ##                         metric threshold    value idx
    ## 1                       max f1  0.132564 0.825000  40
    ## 2                       max f2  0.000002 0.894495  61
    ## 3                 max f0point5  0.132564 0.812808  40
    ## 4                 max accuracy  0.132564 0.787879  40
    ## 5                max precision  0.982938 1.000000   0
    ## 6                   max recall  0.000002 1.000000  61
    ## 7              max specificity  0.982938 1.000000   0
    ## 8             max absolute_mcc  0.132564 0.557317  40
    ## 9   max min_per_class_accuracy  0.837616 0.743590  34
    ## 10 max mean_per_class_accuracy  0.132564 0.774929  40
    ## 
    ## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`

Plotting the test performance's AUC plot shows us approximately how good the predictions are.

``` r
plot(perf)
```

<img src="h2o_files/figure-markdown_github/unnamed-chunk-46-1.png" style="display: block; margin: auto;" />

We also want to know the log loss, MSE and AUC values, as well as other model metrics for the test data:

``` r
h2o.logloss(perf)
```

    ## [1] 1.378809

``` r
h2o.mse(perf)
```

    ## [1] 0.2154912

``` r
h2o.auc(perf)
```

    ## [1] 0.7796771

``` r
head(h2o.metric(perf))
```

    ## Metrics for Thresholds: Binomial metrics as a function of classification thresholds
    ##   threshold       f1       f2 f0point5 accuracy precision   recall
    ## 1  0.982938 0.050000 0.031847 0.116279 0.424242  1.000000 0.025641
    ## 2  0.982811 0.097561 0.063291 0.212766 0.439394  1.000000 0.051282
    ## 3  0.982460 0.095238 0.062893 0.196078 0.424242  0.666667 0.051282
    ## 4  0.982256 0.139535 0.093750 0.272727 0.439394  0.750000 0.076923
    ## 5  0.982215 0.181818 0.124224 0.338983 0.454545  0.800000 0.102564
    ## 6  0.981170 0.177778 0.123457 0.317460 0.439394  0.666667 0.102564
    ##   specificity absolute_mcc min_per_class_accuracy mean_per_class_accuracy
    ## 1    1.000000     0.103203               0.025641                0.512821
    ## 2    1.000000     0.147087               0.051282                0.525641
    ## 3    0.962963     0.033624               0.051282                0.507123
    ## 4    0.962963     0.082188               0.076923                0.519943
    ## 5    0.962963     0.121754               0.102564                0.532764
    ## 6    0.925926     0.048725               0.102564                0.514245
    ##   tns fns fps tps      tnr      fnr      fpr      tpr idx
    ## 1  27  38   0   1 1.000000 0.974359 0.000000 0.025641   0
    ## 2  27  37   0   2 1.000000 0.948718 0.000000 0.051282   1
    ## 3  26  37   1   2 0.962963 0.948718 0.037037 0.051282   2
    ## 4  26  36   1   3 0.962963 0.923077 0.037037 0.076923   3
    ## 5  26  35   1   4 0.962963 0.897436 0.037037 0.102564   4
    ## 6  25  35   2   4 0.925926 0.897436 0.074074 0.102564   5

The confusion matrix alone can be seen with the *h2o.confusionMatrix()* function, but is is also part of the performance summary.

``` r
h2o.confusionMatrix(dl_model, test)
```

The final predictions with probabilities can be extracted with the *h2o.predict()* function. Beware though, that the number of correct and wrong classifications can be slightly different from the confusion matrix above. Here, I combine the predictions with the actual test diagnoses and classes into a dataframe. For plotting I also want to have a column, that tells me whether the predictions were correct. By default, a prediction probability above 0.5 will get scored as a prediction for the respective category. I find it often makes sense to be more stringent with this, though and set a higher threshold. Therefore, I am creating another column with stringent predictions, where I only count predictions that were made with more than 80% probability. Everything that does not fall within this range gets scored as "uncertain". For these stringent predictions, I am also creating a column that tells me whether they were accurate.

``` r
finalRf_predictions <- data.frame(class = as.vector(test$class), actual = as.vector(test$diagnosis), as.data.frame(h2o.predict(object = dl_model, newdata = test)))
```

    ## 
      |                                                                       
      |                                                                 |   0%
      |                                                                       
      |=================================================================| 100%

``` r
finalRf_predictions$accurate <- ifelse(finalRf_predictions$actual == finalRf_predictions$predict, "yes", "no")

finalRf_predictions$predict_stringent <- ifelse(finalRf_predictions$arrhythmia > 0.8, "arrhythmia", 
                                                ifelse(finalRf_predictions$healthy > 0.8, "healthy", "uncertain"))
finalRf_predictions$accurate_stringent <- ifelse(finalRf_predictions$actual == finalRf_predictions$predict_stringent, "yes", 
                                       ifelse(finalRf_predictions$predict_stringent == "uncertain", "na", "no"))
```

``` r
finalRf_predictions %>%
  group_by(actual, predict) %>%
  summarise(n = n())
```

    ## Source: local data frame [4 x 3]
    ## Groups: actual [?]
    ## 
    ##       actual    predict     n
    ##       <fctr>     <fctr> <int>
    ## 1 arrhythmia arrhythmia    16
    ## 2 arrhythmia    healthy    11
    ## 3    healthy arrhythmia     6
    ## 4    healthy    healthy    33

``` r
finalRf_predictions %>%
  group_by(actual, predict_stringent) %>%
  summarise(n = n())
```

    ## Source: local data frame [6 x 3]
    ## Groups: actual [?]
    ## 
    ##       actual predict_stringent     n
    ##       <fctr>             <chr> <int>
    ## 1 arrhythmia        arrhythmia    19
    ## 2 arrhythmia           healthy     6
    ## 3 arrhythmia         uncertain     2
    ## 4    healthy        arrhythmia     8
    ## 5    healthy           healthy    29
    ## 6    healthy         uncertain     2

To get a better overview, I am going to plot the predictions (default and stringent):

``` r
p1 <- finalRf_predictions %>%
  ggplot(aes(x = actual, fill = accurate)) +
    geom_bar(position = "dodge") +
    scale_fill_brewer(palette = "Set1") +
    my_theme() +
    labs(fill = "Were\npredictions\naccurate?",
         title = "Default predictions")

p2 <- finalRf_predictions %>%
  subset(accurate_stringent != "na") %>%
  ggplot(aes(x = actual, fill = accurate_stringent)) +
    geom_bar(position = "dodge") +
    scale_fill_brewer(palette = "Set1") +
    my_theme() +
    labs(fill = "Were\npredictions\naccurate?",
         title = "Stringent predictions")

grid.arrange(p1, p2, ncol = 2)
```

![](h2o_files/figure-markdown_github/unnamed-chunk-53-1.png)

Being more stringent with the prediction threshold slightly reduced the number of errors but not by much.

I also want to know whether there are certain classes of arrhythmia that are especially prone to being misclassified:

``` r
p1 <- subset(finalRf_predictions, actual == "arrhythmia") %>%
  ggplot(aes(x = predict, fill = class)) +
    geom_bar(position = "dodge") +
    my_theme() +
    labs(title = "Prediction accuracy of arrhythmia cases",
         subtitle = "Default predictions",
         x = "predicted to be")

p2 <- subset(finalRf_predictions, actual == "arrhythmia") %>%
  ggplot(aes(x = predict_stringent, fill = class)) +
    geom_bar(position = "dodge") +
    my_theme() +
    labs(title = "Prediction accuracy of arrhythmia cases",
         subtitle = "Stringent predictions",
         x = "predicted to be")

grid.arrange(p1, p2, ncol = 2)
```

![](h2o_files/figure-markdown_github/unnamed-chunk-54-1.png)

There are no obvious biases towards some classes but with the small number of samples for most classes, this is difficult to assess.

<br>

### Final conclusions: How useful is the model?

Most samples were classified correctly, but the total error was not particularly good. Moreover, when evaluating the usefulness of a specific model, we need to keep in mind what we want to achieve with it and which questions we want to answer. If we wanted to deploy this model in a clinical setting, it should assist with diagnosing patients. So, we need to think about what the consequences of wrong classifications would be. Would it be better to optimize for high sensitivity, in this example as many arrhythmia cases as possible get detected - with the drawback that we probably also diagnose a few healthy people? Or do we want to maximize precision, meaning that we could be confident that a patient who got predicted to have arrhythmia does indeed have it, while accepting that a few arrhythmia cases would remain undiagnosed? When we consider stringent predictions, this model correctly classified 19 out of 27 arrhythmia cases, but 6 were misdiagnosed. This would mean that some patients who were actually sick, wouldn't have gotten the correct treatment (if decided solely based on this model). For real-life application, this is obviously not sufficient!

Next week, I'll be trying to improve the model by doing a grid search for hyperparameter tuning.

So, stay tuned... (sorry, couldn't resist ;-))

------------------------------------------------------------------------

<br>

    ## R version 3.3.2 (2016-10-31)
    ## Platform: x86_64-apple-darwin13.4.0 (64-bit)
    ## Running under: macOS Sierra 10.12.3
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## attached base packages:
    ##  [1] stats4    parallel  grid      stats     graphics  grDevices utils    
    ##  [8] datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] reshape2_1.4.2       tidyr_0.6.1          matrixStats_0.51.0  
    ##  [4] pcaGoPromoter_1.18.0 Biostrings_2.42.1    XVector_0.14.0      
    ##  [7] IRanges_2.8.1        S4Vectors_0.12.1     BiocGenerics_0.20.0 
    ## [10] ellipse_0.3-8        gridExtra_2.2.1      ggrepel_0.6.5       
    ## [13] ggplot2_2.2.1        sparklyr_0.5.2       dplyr_0.5.0         
    ## [16] h2o_3.10.3.6         rsparkling_0.1.0    
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] Rcpp_0.12.9          RColorBrewer_1.1-2   plyr_1.8.4          
    ##  [4] zlibbioc_1.20.0      bitops_1.0-6         base64enc_0.1-3     
    ##  [7] tools_3.3.2          digest_0.6.12        memoise_1.0.0       
    ## [10] RSQLite_1.1-2        jsonlite_1.2         evaluate_0.10       
    ## [13] tibble_1.2           gtable_0.2.0         DBI_0.5-1           
    ## [16] yaml_2.1.14          withr_1.0.2          httr_1.2.1          
    ## [19] stringr_1.2.0        knitr_1.15.1         rappdirs_0.3.1      
    ## [22] rprojroot_1.2        Biobase_2.34.0       R6_2.2.0            
    ## [25] AnnotationDbi_1.36.0 rmarkdown_1.3        magrittr_1.5        
    ## [28] backports_1.0.5      scales_0.4.1         htmltools_0.3.5     
    ## [31] assertthat_0.1       colorspace_1.3-2     labeling_0.3        
    ## [34] config_0.2           stringi_1.1.2        RCurl_1.95-4.8      
    ## [37] lazyeval_0.2.0       munsell_0.4.3
