# Customer Churn Prediction

**Goal:** This project predicts which telecom customers are likely to leave
the company (churn) using features such as tenure, contract type and monthly
charges. It trains Logistic Regression on the Telco Customer Churn dataset
(7,043 customers, 21 columns) and compares it with a second model. The best
model is deployed as a Streamlit web app where a user enters customer details
and sees "likely to leave" or "likely to stay", with a probability.

**Target:** the `Churn` column (Yes = left, No = stayed).
**Problem type:** classification (the answer is a label, not a number).
**Key metric:** recall for the "churn" class, plus accuracy.

## Results

I compared three models on 1,409 customers the model had never seen.

| Model | Accuracy | Precision | Recall (churn) |
|---|---|---|---|
| Logistic Regression (plain) | 0.789 | 0.625 | 0.513 |
| Logistic Regression (balanced, final) | 0.732 | 0.497 | 0.789 |
| Random Forest (balanced) | 0.755 | 0.526 | 0.786 |

**In plain words:** Only 26.5% of customers leave, so the plain model played
safe and caught just half of them (recall 0.513). Balancing the classes made
the final model catch about 79% of the customers who really left. The trade-off
is more false alarms: precision is 0.497, so about half of the customers it
flags will not actually leave. This is acceptable because a missed customer
costs the company more than an unnecessary retention offer.

**Biggest clues:** Two-year contracts, longer tenure and tech support point to
staying. Fiber optic internet and electronic check payment point to leaving.

**Note:** The percentage shown in the app is a churn risk score, not the exact
chance that a customer will leave, because the model was balanced.