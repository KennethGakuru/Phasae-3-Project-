# Phasae-3-Project-
**NOTEBOOK STRUCTURE**
1.Overview
2.Business Understanding
3.Data Understanding
4.Data Cleaning
5.Exploratory Data Analysis (**EDA**)
6.Data Preparation
7.Modelling
8.Tuning
9.Recommendation
10.Conclusion

**1.)Overview**
**A.)Introduction**
churn refers to the loss of customers over a specific period. It's a crucial metric for businesses, especially those with subscription models, as it directly impacts their revenue and growth.
**B)Data Overview**

This project focuses on customer data from SyriaTel, a telecommunications company. We have access to a dataset containing 3,333 rows and 21 columns of information about individual customers, including details like service plans, call history, and interactions with customer service.

The columns are: state, account length, area code, phone number, international plan, voice mailplan, number vmail messages, total day minutes, total day calls, total day charge, total eve minutes, total eve calls, total eve charge, total night minutes, total night calls, total night charge, total intl minutes, total intl calls, total intl charge, customer service calls and churn.

**C)PROBLEM STATEMENT**
Customer churn poses a significant financial threat to SyriaTel. When subscribers leave, the company loses recurring revenue and incurs additional costs associated with customer acquisition. To mitigate these challenges, SyriaTel needs a way to predict which customers are at risk of churning before they cancel their service.

**D)Project Objective**
By analyzing the SyriaTel customer data, we aim to build a machine learning model that can effectively predict customer churn. This model will be a valuable tool for SyriaTel, allowing them to proactively identify at-risk customers and implement targeted retention strategies to improve customer satisfaction and loyalty.

**2.Business Understanding:**

**Customer Churn at SyriaTel**

**Problem:**

 Customer churn, the loss of subscribers to competing telecommunications providers, is a significant financial burden for SyriaTel.

**Financial Impact:** Acquiring new customers is significantly more expensive than retaining existing ones. When customers churn, SyriaTel loses recurring revenue from monthly subscriptions, data usage, and other services. This directly impacts the company's profitability.

**Marketing Inefficiency:** Marketing efforts directed towards acquiring new customers can be wasted if existing customers are churning at a high rate. Resources spent on attracting new subscribers could be better utilized to retain satisfied customers who are more likely to generate long-term value.

**Challenge:** 

Completely eliminating customer churn is unrealistic.  Forcing customers to stay with the company is not a sustainable solution.

**Solution:**
 
SyriaTel proposes to develop a Machine Learning (ML) model to predict customer churn.

**Predictive Model:** By analyzing customer data, the ML model will identify subscribers who are at a high risk of churning. This allows SyriaTel to:

**Proactive Customer Care:** The customer care team can proactively reach out to at-risk customers with personalized offers, address their concerns, and improve their overall satisfaction.

**Targeted Marketing:** Marketing efforts can be focused on retaining at-risk customers with targeted campaigns that address their specific needs and preferences. This improves the efficiency and effectiveness of marketing spend.

By implementing a churn prediction model, SyriaTel aims to mitigate customer churn, minimize revenue loss, and optimize marketing efforts for long-term customer retention and growth.

**3.Data Understanding**

**IMPORTING THE RELEVANT LIBRARIES**

**Loading Data**

**4.Data Cleaning.**

**Data Quality:** The dataset is small but has high quality. There's no missing data to pre-process, saving time on data cleaning.

**Class Imbalance:** The dataset is imbalanced, meaning one class (likely churn) is much smaller than the other (likely no churn). This can be problematic for some machine learning algorithms.

**Churn Rate Analysis:** We're interested in analyzing the rate of customer churn (cancellation or inactivity) within the dataset.

# 5.Exploratory Data Analysis (**EDA**)

* What can this tell us about churn?
* Is calling customer service a sign of customer unhappiness/potential churn?
* Are customers in certain areas more likely to churn?
* Other feature engineering and exploration

**Churn Rate Calculation**

**Conclusion**

 The code calculates the churn rate by counting the occurrences of each value in the churn column (True/Yes or False/No) and normalizing the counts to get percentages. This gives you the proportion of customers who churned (True/Yes )[14.5] and those who didn't (False/No)[85.5]. Below is a visualisation of this.

**Churn By State**

![53bc973e-acc2-4fb4-9092-35804335a7f3](https://github.com/KennethGakuru/Phasae-3-Project-/assets/151642075/8b5b562e-78b9-474f-9489-2690a4735bbc)

we group the data by state and churn (True/Yes or False/No), then counts the number of customers in each group. The result is unstacked into a separate column for churn categories, allowing it to be plotted as a stacked bar chart. This visualizes the distribution of churned and non-churned customers across different states.

![f445c662-396d-4e81-9b13-6d63f8d67735](https://github.com/KennethGakuru/Phasae-3-Project-/assets/151642075/c30f301b-9f2d-4e23-ab85-995c8f1c81dc)

![newplot](https://github.com/KennethGakuru/Phasae-3-Project-/assets/151642075/dbd90242-6612-4381-bfc9-a4a26f7dd239)

We create a pie chart showing the distribution of customers across different area codes. It counts the occurrences of each area code.
Plot to visualize the data as pie slices, where each slice represents an area code and its size corresponds to the number of customers associated with it.
Here is a percentage of the distribution of churn based on the ares code:

**25.2% - Area Code 510**

**25.1% - Area Code 408**

**49.7% - Area Code 415**

#**Some Feature engineering**

For this we will adds up all your domestic call minutes (day, evening, night) into a new "total_domestic_minutes" column.

Similar to minutes, it sums your domestic calls (day, evening, night) into a new "total_domestic_calls" column.

It combines all your domestic call charges into a new "total_domestic_charge" column.

Finally, it adds up your total domestic charges and international charges for a final "total_charge".

![1655c412-fc16-4412-bd8b-4904ed2f1428](https://github.com/KennethGakuru/Phasae-3-Project-/assets/151642075/5dd4419c-d8fb-4f33-940e-23d936337804)

![75f77520-7dde-4a73-897b-39a441fa4435](https://github.com/KennethGakuru/Phasae-3-Project-/assets/151642075/0145148c-006f-4964-803

![77d1fe9e-d32f-49d4-a33f-ae6e2c516671](https://github.com/KennethGakuru/Phasae-3-Project-/assets/151642075/ee9d6530-8fc0-4438-bcb3-f6718415d9ea)
a-99dd8df45190)

Overall, there seems to be a seasonal pattern to the number of customer service calls, with peaks in the first, fourth, and sixth months.

The line labeled "Not Churned" shows the number of calls from customers who did not churn (cancel their service) in that month. The number of calls from non-churned customers follows a similar seasonal pattern to the overall call volume, with peaks in the same months.

The line labeled "Churned" shows the number of calls from customers who churned in that month. The number of calls from churned customers is generally lower than the number of calls from non-churned customers. However, there is a spike in churned customers in month four, which coincides with a peak in overall call volume.

**Conclusion**

-Most customers who contacted customer service made 0 to 4 calls.

-There are a higher proportion of non-churned customers than churned customers across all call counts.

-A slightly higher proportion of customers who churned made 4 calls compared to non-churned customers.

**Identify trends:** 

Look for patterns in the scatter plots, such as clusters of points, linear relationships, or non-linear relationships. These patterns may indicate how different variables are related to customer churn.

**Focus on churned customers:**

Pay close attention to the subplots that show the relationship between a specific variable and the churn label. See if there are any trends that differentiate churned customers from non-churned customers.

**Consider multiple variables:** 

Look for relationships between multiple variables that might contribute to churn. For example, you might see that customers who make a high number of customer service calls and have low satisfaction scores are more likely to churn

**SPLITTING THE DATA SET**

**SPLITTING THE DATA SET**

**Logistic Regression Model**



              precision    recall  f1-score   support

       False       0.88      0.97      0.93       857
        True       0.59      0.22      0.32       143

    accuracy                           0.87      1000
   macro avg       0.74      0.60      0.63      1000
weighted avg       0.84      0.87      0.84      1000


**Evaluation Metrics**


**Precision: Proportion of predicted positives that were actually positive.

High precision for "False" (0.88) indicates the model rarely mistakes negative cases as positive.

Low precision for "True" (0.59) indicates the model often misses actual positive cases.

Recall: Proportion of actual positives that were correctly identified.

High recall for "False" (0.97) indicates the model effectively identifies most negative cases.

Low recall for "True" (0.22) indicates the model misses many actual positive cases (high false negatives).

F1-Score: Harmonic mean of precision and recall, combining both metrics into a single score.

A balanced F1-score is desirable, but the imbalance here reflects the trade-off between precision and recall.

Support: Number of samples in each class ("False" and "True").
Overall Accuracy:

Accuracy (0.87) indicates the model correctly classified 87% of the 1000 samples**

**K-FOLD VALIDATION**

**Conclusion**

The logistic regression model (lr_regression) achieved an average precision of 51.6% on the cross-validation test. While this might seem like a moderate accuracy, it's not ideal for a churn prediction model 

However, we also need to check on the precision of the model. This is because:

Precision focuses on identifying true positives: In churn prediction, we care more about accurately identifying customers who will churn (positives) and taking action to retain them.

Low precision for "True" (0.59) indicates the model often misses actual positive cases.

**Decision Tree**

**8.Tuning**

### Hyperparameter Tuning of Decision Trees for Regression

0.8501857083511273


# Evaluate the model's r2 score on the training data for reference
dt_tuned_train_score = dt_tuned.score(X_train, y_train)
print("Training r-squared:", dt_tuned_train_score)

Training r-squared: 0.8233167074793086

              precision    recall  f1-score   support

       False       0.88      0.97      0.93       857
        True       0.59      0.22      0.32       143

    accuracy                           0.87      1000
   macro avg       0.74      0.60      0.63      1000
weighted avg       0.84      0.87      0.84      1000

# Get the best model from GridSearchCV
dt_tuned = grid_search.best_estimator_

# Evaluate the tuned model's performance
y_pred = dt_tuned.predict(X_test)


# Print the best hyperparameters for reference
print("Best hyperparameters:", grid_search.best_params_)

Best hyperparameters: {'criterion': 'squared_error', 'max_depth': 5, 'min_samples_split': 10}

# Create a second decision tree model
dt_tuned_1 = DecisionTreeRegressor(criterion='squared_error', max_depth=5, min_samples_split=10)


# Fit the new model on the training data
dt_tuned_1.fit(X_train, y_train)

# Testing out the model's r2 score on the training data overall
dt_tuned_train_score = dt_tuned_1.score(X_train, y_train)
dt_tuned_train_score

0.8233167074793086

# Make predictions on the testing data
y_pred = dt_tuned_1.predict(X_test)


dt_tuned_cv = cross_val_score(dt_tuned_1, X_train, y_train, cv = 5)


dt_tuned_cv.mean()

0.8065929141409276

from sklearn.metrics import r2_score

y_pred = dt_tuned_1.predict(X_train)
train_r2 = r2_score(y_train, y_pred)  # Use r2_score for regression

print("Training R-squared:", train_r2)

Training R-squared: 0.8233167074793086

# Evaluate the tuned model's performance on the testing data
y_pred = dt_tuned_1.predict(X_test)
test_r2 = r2_score(y_test, y_pred)  # Use r2_score for regression


print("Testing R-squared:", test_r2)

Testing R-squared: 0.8596578731245792


from sklearn.metrics import mean_squared_error

y_pred = dt_tuned_1.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

Mean Squared Error: 0.017199067990709686


from sklearn.preprocessing import Binarizer

# Define a threshold (e.g., 0.5 for probability)
threshold = 0.5

# Binarize predictions
y_pred_binary = Binarizer(threshold=threshold).fit_transform(y_pred.reshape(-1, 1))

# Use classification metrics on the binarized predictions
precision_true = precision_score(y_test, y_pred_binary, pos_label=1)
recall_true = recall_score(y_test, y_pred_binary, pos_label=1)

print("Precision (True):", precision_true)
print("Recall (True):", recall_true)

Precision (True): 0.9921875
Recall (True): 0.8881118881118881

**Based on the analysis:**


**High Accuracy:** The model achieved a Training R-squared of 0.82 and Testing R-squared of 0.86, indicating a good fit on both training data and generalizability to unseen data.

**Effective Churn Identification:** The model has a Precision (True) of 0.98, meaning it accurately identifies most churn events.

**Room for Improvement in Recall:** While the Recall (True) of 0.89 is good, there's an opportunity to capture a higher percentage of actual churn events (false negatives).

**9.Recommendation**

**A).Targeted Marketing and Customer Service:**
Leverage the churn prediction model to identify customers at high risk of churn. This allows SyriaTel to:

**-Targeted Marketing:** 

Design and deliver personalized marketing campaigns to high-risk customers, potentially offering incentives or highlighting features that address their specific needs. This can re-engage these customers and encourage them to stay.

**-Proactive Customer Service:** 

Proactively reach out to high-risk customers with personalized support or retention offers. Understanding their potential reasons for churn can help address concerns and improve customer satisfaction.

**B).Develop Customized Service Models:**

Analyze the characteristics of high-risk customers identified by the model. This can help SyriaTel develop customized service models tailored to different customer segments. For example, high-value or high-engagement customers might benefit from dedicated account managers or premium service packages.

**C).Model Improvement and Monitoring:**

**Hyperparameter Tuning:**

Further refine the model by exploring hyperparameter tuning techniques to potentially improve its generalization and reduce false negatives (missed churn events).

**Continuous Monitoring:**

Regularly monitor the model's performance over time and retrain it with new data to ensure it remains accurate and adapts to changing customer behavior.

**10.Conclusion**

**Reduced Customer Churn:**

Retaining existing customers is significantly cheaper than acquiring new ones. Reduced churn translates to direct cost savings for SyriaTel.

**Improved Customer Satisfaction:**
 
Proactive engagement and personalized service can improve customer satisfaction and loyalty, further reducing churn.

**Increased Revenue:**

Retained customers are more likely to spend more, leading to increased revenue for SyriaTel.







