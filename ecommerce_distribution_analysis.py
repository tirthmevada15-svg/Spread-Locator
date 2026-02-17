import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom, poisson, lognorm, powerlaw, norm, boxcox
import scipy.stats as stats

print("\n===== LOADING DATASET =====")

# Load dataset
df = pd.read_excel("spread_locator_dataset.xlsx")

print(df.head())
print("\nColumns:", df.columns)

print("\n===== DATA CLEANING =====")

df['transaction_date'] = pd.to_datetime(df['transaction_date'])

df['success'] = df['transaction_status'].map({'Success':1,'Fail':0})

print(df.describe())

print("\n===== BERNOULLI DISTRIBUTION =====")

p_success = df['success'].mean()
print("Probability of Successful Transaction:", round(p_success,4))

print("\n===== BINOMIAL DISTRIBUTION =====")

n = int(df['transaction_count'].max())
k = int(df['transaction_count'].mean())

binomial_prob = binom.pmf(k, n, p_success)
print("Binomial Probability:", binomial_prob)

print("\n===== POISSON DISTRIBUTION =====")

daily_orders = df.groupby('transaction_date').size()
lam = daily_orders.mean()

poisson_prob = poisson.pmf(int(lam), lam)
print("Average daily orders:", lam)
print("Poisson Probability:", poisson_prob)

print("\n===== LOG-NORMAL DISTRIBUTION =====")

amounts = df['transaction_amount']

shape, loc, scale = lognorm.fit(amounts)
print("Shape:", shape, "Loc:", loc, "Scale:", scale)

print("\n===== POWER LAW DISTRIBUTION =====")

params = powerlaw.fit(amounts)
print("Power Law Parameters:", params)

print("\n===== Q-Q PLOT =====")

plt.figure(figsize=(6,6))
stats.probplot(amounts, dist="norm", plot=plt)
plt.title("Q-Q Plot for Transaction Amount")
plt.show()

print("\n===== BOX-COX TRANSFORMATION =====")

positive_amounts = amounts + 1
transformed, lam_boxcox = boxcox(positive_amounts)

print("Lambda value:", lam_boxcox)

plt.figure(figsize=(6,4))
sns.histplot(transformed, kde=True)
plt.title("After Box-Cox Transformation")
plt.show()

print("\n===== Z-SCORE PROBABILITY =====")

mean = amounts.mean()
std = amounts.std()

z = (5000 - mean)/std
probability = 1 - norm.cdf(z)

print("Mean:", mean)
print("Std:", std)
print("Z-score:", z)
print("Probability transaction > 5000:", probability)

print("\n===== PDF & CDF =====")

x = np.linspace(min(amounts), max(amounts), 200)

pdf = norm.pdf(x, mean, std)
cdf = norm.cdf(x, mean, std)

plt.figure(figsize=(8,5))
plt.plot(x, pdf, label="PDF")
plt.plot(x, cdf, label="CDF")
plt.legend()
plt.title("PDF vs CDF")
plt.xlabel("Transaction Amount")
plt.ylabel("Probability")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(amounts, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()

print("\n===== FINAL CONCLUSION =====")
print("Transaction amounts follow a LOG-NORMAL distribution (right-skewed).")
print("Few customers spend large amounts while majority spend small amounts.")
print("This is typical E-commerce customer behavior.")

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")
