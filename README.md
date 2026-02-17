# E-Commerce Transaction Probability Distribution Analysis

## Project Overview

This project analyzes customer transaction behavior of an e-commerce platform using **statistical probability distributions** and **data transformations**.
The goal is to understand purchasing patterns, model transaction behavior, and extract business insights for decision making.

The analysis applies theoretical statistics concepts to real-world transactional data using Python.


## Objectives

* Understand distribution of transaction amounts
* Check whether data follows normal distribution
* Model customer behavior using probability distributions
* Calculate probability of high-value purchases
* Stabilize skewed financial data
* Generate business insights from statistical results

## Dataset Description

| Column             | Description                |
| ------------------ | -------------------------- |
| transaction_id     | Unique ID of transaction   |
| customer_id        | Unique customer identifier |
| transaction_amount | Amount spent (₹)           |
| transaction_date   | Date of purchase           |
| transaction_count  | Weekly purchase count      |
| region             | Customer region            |
| transaction_status | Success / Fail             |

## Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib
* Seaborn
* Statsmodels

---

## Statistical Concepts Implemented

### Distributions

* Bernoulli Distribution (Transaction success)
* Binomial Distribution (Weekly purchases)
* Poisson Distribution (Daily orders)
* Log-Normal Distribution (Transaction amounts)
* Power Law Distribution (High-spending customers)

### Statistical Methods

* Q-Q Plot (Normality test)
* Box-Cox Transformation
* Z-Score Probability
* PDF & CDF Analysis

## How to Run

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```
### Place dataset

Put file inside project folder:
transactions.xlsx

## Key Findings

* Transaction amounts are right-skewed
* Data follows Log-Normal distribution
* Few customers generate most revenue (Power Law behavior)
* Daily orders follow Poisson process
* High-value purchases are rare events

## Business Insights

* Focus on VIP customer retention
* Promote mid-range products
* Improve payment success rate
* Offer EMI/discounts for expensive products
* Use median instead of average revenue metrics
* 
## Final Conclusion

E-commerce transactions do not follow normal distribution.
Customer spending behavior follows **Log-Normal and Power-Law patterns**, meaning a small percentage of customers contributes most of the revenue.
Understanding these patterns helps optimize marketing strategies, inventory planning, and revenue forecasting.
