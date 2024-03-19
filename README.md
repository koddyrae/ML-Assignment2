# ML-Assignment2 
Task 2 Findings

The purpose of this model is to predict the quality of wine based on various attributes using regression analysis. 

Data Preprocessing: 

- Creates a feature called "alcohol_cat" based on the "alcohol" column in wine_data. It bins the alcohol levels into five categories using the pd.cut function, with predefined bins and corresponding labels.
We studied the correlation matrix to find out which features have the strongest correlation to wine quality. Using said information, we identified and removed Irrelevant features with the lowest correlation ('sulphates', 'free sulfur dioxide', 'citric acid'). These features were removed because they had the lowest positive or negative correlation to the quality feature. 

Correlation Analysis
- We created a correlation matrix to identify the features with strongest correlation to quality. The correlation score also includes negative correlation: 

As we can see, the alcohol correlation was the strongest followed by pH. The worst performing correlation being sulphates, free sulfur dioxide and citric acid. As such, these irrelevant features are removed. 
Data Split
The dataset is then split into training and testing sets using stratified sampling based on the "alcohol_cat" column. The test set makes up 20% of the total dataset.




Model Experimentation and Evaluation 
Oversampling
- To fix the unbalanced dataset, we applied oversampling to the data using RandomOverSampler. The Sampler randomly analyzes the data to achieve even distribution. Since there is a small number of low quality wine samples, using an unbalanced dataset would produce a biased model and could cause overfitting problems. 
Model training 
- To maximize performance, we trained and evaluated many models such as, Linear Regression, Decision Tree, Random Forest, Bayesian Ridge, Gradient boosting , KNeighbors, Ridge, Lasso, ElasticNet, Huber, PassiveAggresive, RANSAC, and Support Vector Regressor. 
- The models gave us a wide array of results. Ultimately, we decided to utilize the Random Forest Regressor model.
- We decided to choose the model that gave us the best overall scores on the MSE, MAE, R2, and CV_RMSE since that would mean that it is the model that best fits our data.
- The RandomForestRegressor model had the best overall performing data across all of the metrics that we were evaluating from.
Hyperparameter Tuning 
- For fine tuning the parameters, we decide to use GridSearchCV from SciKit-Learn

Running Final Model and Evaluation
Lastly, we calculated the RMSE following the hypertunning. This allows us to estimate our models accuracy. It is worth noting that, because the wine quality values did not have a large range, having an RMSE of 0.673 shows us that the model is fairly accurate and suggests that we are within a good range for the quality that it predicts.   


