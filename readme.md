![COUR_IPO.png](vertopal_bad73b54b05e4bdcb74fd894d0d06097/COUR_IPO.png)

# Welcome to the Data Science Coding Challange!

Test your skills in a real-world coding challenge. Coding Challenges
provide CS & DS Coding Competitions with Prizes and achievement badges!

CS & DS learners want to be challenged as a way to evaluate if they're
job ready. So, why not create fun challenges and give winners something
truly valuable such as complimentary access to select Data Science
courses, or the ability to receive an achievement badge on their
Coursera Skills Profile - highlighting their performance to recruiters.

## Introduction

In this challenge, you\'ll get the opportunity to tackle one of the most
industry-relevant maching learning problems with a unique dataset that
will put your modeling skills to the test. Subscription services are
leveraged by companies across many industries, from fitness to video
streaming to retail. One of the primary objectives of companies with
subscription services is to decrease churn and ensure that users are
retained as subscribers. In order to do this efficiently and
systematically, many companies employ machine learning to predict which
users are at the highest risk of churn, so that proper interventions can
be effectively deployed to the right audience.

In this challenge, we will be tackling the churn prediction problem on a
very unique and interesting group of subscribers on a video streaming
service!

Imagine that you are a new data scientist at this video streaming
company and you are tasked with building a model that can predict which
existing subscribers will continue their subscriptions for another
month. We have provided a dataset that is a sample of subscriptions that
were initiated in 2021, all snapshotted at a particular date before the
subscription was cancelled. Subscription cancellation can happen for a
multitude of reasons, including:

-   the customer completes all content they were interested in, and no
    longer need the subscription
-   the customer finds themselves to be too busy and cancels their
    subscription until a later time
-   the customer determines that the streaming service is not the best
    fit for them, so they cancel and look for something better suited

Regardless the reason, this video streaming company has a vested
interest in understanding the likelihood of each individual customer to
churn in their subscription so that resources can be allocated
appropriately to support customers. In this challenge, you will use your
machine learning toolkit to do just that!

## Understanding the Datasets

### Train vs. Test {#train-vs-test}

In this competition, you'll gain access to two datasets that are samples
of past subscriptions of a video streaming platform that contain
information about the customer, the customers streaming preferences, and
their activity in the subscription thus far. One dataset is titled
`train.csv` and the other is titled `test.csv`.

`train.csv` contains 70% of the overall sample (243,787 subscriptions to
be exact) and importantly, will reveal whether or not the subscription
was continued into the next month (the "ground truth").

The `test.csv` dataset contains the exact same information about the
remaining segment of the overall sample (104,480 subscriptions to be
exact), but does not disclose the "ground truth" for each subscription.
It's your job to predict this outcome!

Using the patterns you find in the `train.csv` data, predict whether the
subscriptions in `test.csv` will be continued for another month, or not.

### Dataset descriptions

Both `train.csv` and `test.csv` contain one row for each unique
subscription. For each subscription, a single observation (`CustomerID`)
is included during which the subscription was active.

In addition to this identifier column, the `train.csv` dataset also
contains the target label for the task, a binary column `Churn`.

Besides that column, both datasets have an identical set of features
that can be used to train your model to make predictions. Below you can
see descriptions of each feature. Familiarize yourself with them so that
you can harness them most effectively for this machine learning task!

``` python
import pandas as pd
data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions
```

| Column_name            | Column_type | Data_type | Description                                                      |
|------------------------|-------------|-----------|------------------------------------------------------------------|
| AccountAge             | Feature     | integer   | The age of the user's account in months.                          |
| MonthlyCharges         | Feature     | float     | The amount charged to the user on a monthly basis.               |
| TotalCharges           | Feature     | float     | The total charges incurred by the user over the account's lifetime.|
| SubscriptionType       | Feature     | object    | The type of subscription chosen by the user (Basic, Standard, or Premium).|
| PaymentMethod          | Feature     | string    | The method of payment used by the user.                          |
| PaperlessBilling       | Feature     | string    | Indicates whether the user has opted for paperless billing (Yes or No).|
| ContentType            | Feature     | string    | The type of content preferred by the user (Movies, TV Shows, or Both).|
| MultiDeviceAccess      | Feature     | string    | Indicates whether the user has access to the service on multiple devices (Yes or No).|
| DeviceRegistered       | Feature     | string    | The type of device registered by the user (TV, Mobile, Tablet, or Computer).|
| ViewingHoursPerWeek    | Feature     | float     | The number of hours the user spends watching content per week.   |
| AverageViewingDuration | Feature     | float     | The average duration of each viewing session in minutes.         |
| ContentDownloadsPerMonth | Feature   | integer   | The number of content downloads by the user per month.           |
| GenrePreference        | Feature     | string    | The preferred genre of content chosen by the user.               |
| UserRating             | Feature     | float     | The user's rating for the service on a scale of 1 to 5.          |
| SupportTicketsPerMonth | Feature     | integer   | The number of support tickets raised by the user per month.      |
| Gender                 | Feature     | string    | The gender of the user (Male or Female).                         |
| WatchlistSize          | Feature     | float     | The number of items in the user's watchlist.                     |
| ParentalControl        | Feature     | string    | Indicates whether parental control is enabled for the user (Yes or No).|
| SubtitlesEnabled       | Feature     | string    | Indicates whether subtitles are enabled for the user (Yes or No).|
| CustomerID             | Identifier  | string    | A unique identifier for each customer.                           |
| Churn                  | Target      | integer   | The target variable indicating whether a user has churned or not (1 for churned, 0 for not churned).|

## How to Submit your Predictions to Coursera

Submission Format:

In this notebook you should follow the steps below to explore the data,
train a model using the data in `train.csv`, and then score your model
using the data in `test.csv`. Your final submission should be a
dataframe (call it `prediction_df` with two columns and exactly 104,480
rows (plus a header row). The first column should be `CustomerID` so
that we know which prediction belongs to which observation. The second
column should be called `predicted_probability` and should be a numeric
column representing the **likellihood that the subscription will
churn**.

Your submission will show an error if you have extra columns (beyond
`CustomerID` and `predicted_probability`) or extra rows. The order of
the rows does not matter.

The naming convention of the dataframe and columns are critical for our
autograding, so please make sure to use the exact naming conventions of
`prediction_df` with column names `CustomerID` and
`predicted_probability`!

To determine your final score, we will compare your
`predicted_probability` predictions to the source of truth labels for
the observations in `test.csv` and calculate the [ROC
AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
We choose this metric because we not only want to be able to predict
which subscriptions will be retained, but also want a well-calibrated
likelihood score that can be used to target interventions and support
most accurately.
:::

::: {.cell .markdown}
## Import Python Modules

First, import the primary modules that will be used in this project.
Remember as this is an open-ended project please feel free to make use
of any of your favorite libraries that you feel may be useful for this
challenge. For example some of the following popular packages may be
useful:

-   pandas
-   numpy
-   Scipy
-   Scikit-learn
-   keras
-   maplotlib
-   seaborn
-   etc, etc
:::

::: {.cell .code execution_count="2"}
``` python
# Import required packages

# Data packages
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
```
:::

::: {.cell .code execution_count="3"}
``` python
# Import any other packages you may want to use
import scipy.stats
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
```
:::

::: {.cell .markdown}
## Load the Data

Let\'s start by loading the dataset `train.csv` into a dataframe
`train_df`, and `test.csv` into a dataframe `test_df` and display the
shape of the dataframes.
:::

::: {.cell .code execution_count="4"}
``` python
train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()
```

::: {.output .stream .stdout}
    train_df Shape: (243787, 21)
:::

::: {.output .execute_result execution_count="4"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>SubscriptionType</th>
      <th>PaymentMethod</th>
      <th>PaperlessBilling</th>
      <th>ContentType</th>
      <th>MultiDeviceAccess</th>
      <th>DeviceRegistered</th>
      <th>ViewingHoursPerWeek</th>
      <th>...</th>
      <th>ContentDownloadsPerMonth</th>
      <th>GenrePreference</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>Gender</th>
      <th>WatchlistSize</th>
      <th>ParentalControl</th>
      <th>SubtitlesEnabled</th>
      <th>CustomerID</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>11.055215</td>
      <td>221.104302</td>
      <td>Premium</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>Both</td>
      <td>No</td>
      <td>Mobile</td>
      <td>36.758104</td>
      <td>...</td>
      <td>10</td>
      <td>Sci-Fi</td>
      <td>2.176498</td>
      <td>4</td>
      <td>Male</td>
      <td>3</td>
      <td>No</td>
      <td>No</td>
      <td>CB6SXPNVZA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>5.175208</td>
      <td>294.986882</td>
      <td>Basic</td>
      <td>Credit card</td>
      <td>Yes</td>
      <td>Movies</td>
      <td>No</td>
      <td>Tablet</td>
      <td>32.450568</td>
      <td>...</td>
      <td>18</td>
      <td>Action</td>
      <td>3.478632</td>
      <td>8</td>
      <td>Male</td>
      <td>23</td>
      <td>No</td>
      <td>Yes</td>
      <td>S7R2G87O09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>73</td>
      <td>12.106657</td>
      <td>883.785952</td>
      <td>Basic</td>
      <td>Mailed check</td>
      <td>Yes</td>
      <td>Movies</td>
      <td>No</td>
      <td>Computer</td>
      <td>7.395160</td>
      <td>...</td>
      <td>23</td>
      <td>Fantasy</td>
      <td>4.238824</td>
      <td>6</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>EASDC20BDT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>7.263743</td>
      <td>232.439774</td>
      <td>Basic</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>Tablet</td>
      <td>27.960389</td>
      <td>...</td>
      <td>30</td>
      <td>Drama</td>
      <td>4.276013</td>
      <td>2</td>
      <td>Male</td>
      <td>24</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>NPF69NT69N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>16.953078</td>
      <td>966.325422</td>
      <td>Premium</td>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>TV</td>
      <td>20.083397</td>
      <td>...</td>
      <td>20</td>
      <td>Comedy</td>
      <td>3.616170</td>
      <td>4</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>4LGYPK7VOL</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="5" scrolled="true"}
``` python
test_df = pd.read_csv("test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()
```

::: {.output .stream .stdout}
    test_df Shape: (104480, 20)
:::

::: {.output .execute_result execution_count="5"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>SubscriptionType</th>
      <th>PaymentMethod</th>
      <th>PaperlessBilling</th>
      <th>ContentType</th>
      <th>MultiDeviceAccess</th>
      <th>DeviceRegistered</th>
      <th>ViewingHoursPerWeek</th>
      <th>AverageViewingDuration</th>
      <th>ContentDownloadsPerMonth</th>
      <th>GenrePreference</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>Gender</th>
      <th>WatchlistSize</th>
      <th>ParentalControl</th>
      <th>SubtitlesEnabled</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>17.869374</td>
      <td>679.036195</td>
      <td>Premium</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>TV</td>
      <td>29.126308</td>
      <td>122.274031</td>
      <td>42</td>
      <td>Comedy</td>
      <td>3.522724</td>
      <td>2</td>
      <td>Male</td>
      <td>23</td>
      <td>No</td>
      <td>No</td>
      <td>O1W6BHP6RM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>9.912854</td>
      <td>763.289768</td>
      <td>Basic</td>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>TV</td>
      <td>36.873729</td>
      <td>57.093319</td>
      <td>43</td>
      <td>Action</td>
      <td>2.021545</td>
      <td>2</td>
      <td>Female</td>
      <td>22</td>
      <td>Yes</td>
      <td>No</td>
      <td>LFR4X92X8H</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>15.019011</td>
      <td>75.095057</td>
      <td>Standard</td>
      <td>Bank transfer</td>
      <td>No</td>
      <td>TV Shows</td>
      <td>Yes</td>
      <td>Computer</td>
      <td>7.601729</td>
      <td>140.414001</td>
      <td>14</td>
      <td>Sci-Fi</td>
      <td>4.806126</td>
      <td>2</td>
      <td>Female</td>
      <td>22</td>
      <td>No</td>
      <td>Yes</td>
      <td>QM5GBIYODA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88</td>
      <td>15.357406</td>
      <td>1351.451692</td>
      <td>Standard</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>Both</td>
      <td>Yes</td>
      <td>Tablet</td>
      <td>35.586430</td>
      <td>177.002419</td>
      <td>14</td>
      <td>Comedy</td>
      <td>4.943900</td>
      <td>0</td>
      <td>Female</td>
      <td>23</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>D9RXTK2K9F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>91</td>
      <td>12.406033</td>
      <td>1128.949004</td>
      <td>Standard</td>
      <td>Credit card</td>
      <td>Yes</td>
      <td>TV Shows</td>
      <td>Yes</td>
      <td>Tablet</td>
      <td>23.503651</td>
      <td>70.308376</td>
      <td>6</td>
      <td>Drama</td>
      <td>2.846880</td>
      <td>6</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>ENTCCHR1LR</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
## Explore, Clean, Validate, and Visualize the Data (optional)

Feel free to explore, clean, validate, and visualize the data however
you see fit for this competition to help determine or optimize your
predictive model. Please note - the final autograding will only be on
the accuracy of the `prediction_df` predictions.
:::

::: {.cell .code execution_count="6" scrolled="false"}
``` python
# your code here (optional)
test_df = test_df.dropna()
train_df = train_df.dropna()

test_df = test_df.drop_duplicates()
train_df = train_df.drop_duplicates()


# Plot the distribution of the target variable 'Churn'
sns.countplot(data=train_df, x='Churn')
plt.title('Distribution of Churn')
plt.show()

# Finding distributions of the categorical features
categorical_features = ['SubscriptionType', 'PaymentMethod', 'ContentType', 'DeviceRegistered', 'GenrePreference', 'Gender']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

for i, col in enumerate(categorical_features):
    train_df[col].value_counts().plot(kind='bar', ax=axs[i])
    axs[i].set_title(f'Distribution of {col}')
    axs[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/86eade401a63b1004817af86a4c675be010d45a8.png)
:::

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/5fb81e036de2a5764090ec42d46dca29525b9f6d.png)
:::
:::

::: {.cell .markdown}
The distributions of the categorical features are even and looks
uniform.

Below is zoom in on the y-axis.
:::

::: {.cell .code execution_count="7"}
``` python
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

for i, col in enumerate(categorical_features):
    train_df[col].value_counts().plot(kind='bar', ax=axs[i], logy=True)
    axs[i].set_title(f'Distribution of {col}')
    axs[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/d5d381bd094677adb613cfd052a853862dfaf83b.png)
:::
:::

::: {.cell .code execution_count="8"}
``` python
# Set the style of seaborn plot
sns.set(style="whitegrid")

# Function to plot a boxplot
def plot_box(data, x_column, y_column, title, xlabel, ylabel, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_column, y=y_column, data=data)
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
```
:::

::: {.cell .code execution_count="9" scrolled="false"}
``` python
# Plot the distribution of some numerical features
numerical_features = ['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek']

# Plot histograms for each numerical feature
for i, col in enumerate(numerical_features):
    plot_box(train_df, 'Churn', col, f'{col} vs Churn', 'Churn', col)

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/297ddff575ddc9cb692d664fbe0794d15eccb212.png)
:::

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/3fd91d19dfe3bf10af6442476d43a36acc810287.png)
:::

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/69009ee92d7ffb401fb997ca9099d2b03494cd23.png)
:::

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/4fe3b98d1e1ab96164f23fe993634ab1dd563c19.png)
:::
:::

::: {.cell .code execution_count="10" scrolled="false"}
``` python
# Select numerical columns
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns

# Compute the correlation matrix
corr = train_df[numerical_cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/065f6556257c43e723e0e954975544f640909211.png)
:::
:::

::: {.cell .markdown}
From the heatmap above, there are several insights we can see:

1.  The correlation between \'AccountAge\' and \'TotalCharges\' is
    strongly positive (0.82). This is obvious because the longer the
    customer own the account, the more they are paying over time.
2.  \'TotalCharges\' and \'MonthlyCharges\' has positive correlation
    (0.5). The recurring charges contributes to the total charge.
3.  \'AccountAge\', \'TotalCharges\', \'ViewingHoursPerWeek\',
    \'AverageViewingDuration\', \'ContentDownloadsPermonths\' all have
    negative correlation with \'Churn\' because the longer the customer
    own the account, the longer the spend watching, and the more they
    pay would result in them likely to cancel the subscription.
:::

::: {.cell .markdown}
## Make predictions (required)

Remember you should create a dataframe named `prediction_df` with
exactly 104,480 entries plus a header row attempting to predict the
likelihood of churn for subscriptions in `test_df`. Your submission will
throw an error if you have extra columns (beyond `CustomerID` and
`predicted_probaility`) or extra rows.

The file should have exactly 2 columns: `CustomerID` (sorted in any
order) `predicted_probability` (contains your numeric predicted
probabilities between 0 and 1, e.g. from
`estimator.predict_proba(X, y)[:, 1]`)

The naming convention of the dataframe and columns are critical for our
autograding, so please make sure to use the exact naming conventions of
`prediction_df` with column names `CustomerID` and
`predicted_probability`!
:::

::: {.cell .markdown}
### Example prediction submission:

The code below is a very naive prediction method that simply predicts
churn using a Dummy Classifier. This is used as just an example showing
the submission format required. Please change/alter/delete this code
below and create your own improved prediction methods for generating
`prediction_df`.
:::

::: {.cell .markdown}
**PLEASE CHANGE CODE BELOW TO IMPLEMENT YOUR OWN PREDICTIONS**
:::

::: {.cell .markdown}
# One Hot Encoding and Standard Scaler
:::

::: {.cell .markdown}
Let\'s start with encoding the categorical variables. We\'ll use one-hot
encoding, which is a common method for converting categorical variables
into a format that works better with classification and regression
algorithms. It creates new binary columns for each category/label
present in the original columns
:::

::: {.cell .code execution_count="11"}
``` python
### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# Fit a dummy classifier on the feature columns in train_df:
# dummy_clf = DummyClassifier(strategy="stratified")
# dummy_clf.fit(train_df.drop(['CustomerID', 'Churn'], axis=1), train_df.Churn)

# Initialize one-hot-encoder
ohe = OneHotEncoder(sparse=False, drop='first')

# Select categorical columns
categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('CustomerID')  # We won't one-hot encode the CustomerID

# Fit and transform the one hot encoder on the data
train_encoded = ohe.fit_transform(train_df[categorical_columns])

# Get feature names
feature_names = ohe.get_feature_names(categorical_columns)

# Create a DataFrame with the one-hot encoded categorical variables
train_encoded = pd.DataFrame(train_encoded, columns=feature_names)

# Concatenate the original numerical columns and the one-hot encoded columns
train_data_encoded = pd.concat([train_df.select_dtypes(include=['int64', 'float64']), train_encoded], axis=1)

train_data_encoded.head()
```

::: {.output .execute_result execution_count="11"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>ViewingHoursPerWeek</th>
      <th>AverageViewingDuration</th>
      <th>ContentDownloadsPerMonth</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>WatchlistSize</th>
      <th>Churn</th>
      <th>...</th>
      <th>DeviceRegistered_Mobile</th>
      <th>DeviceRegistered_TV</th>
      <th>DeviceRegistered_Tablet</th>
      <th>GenrePreference_Comedy</th>
      <th>GenrePreference_Drama</th>
      <th>GenrePreference_Fantasy</th>
      <th>GenrePreference_Sci-Fi</th>
      <th>Gender_Male</th>
      <th>ParentalControl_Yes</th>
      <th>SubtitlesEnabled_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>11.055215</td>
      <td>221.104302</td>
      <td>36.758104</td>
      <td>63.531377</td>
      <td>10</td>
      <td>2.176498</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>5.175208</td>
      <td>294.986882</td>
      <td>32.450568</td>
      <td>25.725595</td>
      <td>18</td>
      <td>3.478632</td>
      <td>8</td>
      <td>23</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>73</td>
      <td>12.106657</td>
      <td>883.785952</td>
      <td>7.395160</td>
      <td>57.364061</td>
      <td>23</td>
      <td>4.238824</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>7.263743</td>
      <td>232.439774</td>
      <td>27.960389</td>
      <td>131.537507</td>
      <td>30</td>
      <td>4.276013</td>
      <td>2</td>
      <td>24</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>16.953078</td>
      <td>966.325422</td>
      <td>20.083397</td>
      <td>45.356653</td>
      <td>20</td>
      <td>3.616170</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
We have successfully one-hot encoded the categorical variables in the
training data. We can see that new binary columns have been created for
each category in each of the original categorical variables. For
example, for SubscriptionType, we now have SubscriptionType_Premium and
SubscriptionType_Standard, where a 1 indicates that the original
SubscriptionType was \'Premium\' or \'Standard\', respectively. The
original \'Basic\' category is represented by a 0 in both of these new
columns due to the drop=\'first\' argument in the OneHotEncoder.

Next, let\'s scale the numerical variables. For this, we can use
standard scaling, which transforms the variables to have a mean of 0 and
a standard deviation of 1. This ensures that all numerical features have
the same scale, which is important for many machine learning algorithms.
:::

::: {.cell .code execution_count="12"}
``` python
# Initialize standard scaler
scaler = StandardScaler()

# Select numerical columns
numerical_columns = train_data_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_columns.remove('Churn')  # We won't scale the target variable

# Fit the scaler on the training data
scaler.fit(train_data_encoded[numerical_columns])

# Scale the training data
train_data_encoded[numerical_columns] = scaler.transform(train_data_encoded[numerical_columns])

# Show the first few rows of the preprocessed training data
train_data_encoded.head()
```

::: {.output .execute_result execution_count="12"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>ViewingHoursPerWeek</th>
      <th>AverageViewingDuration</th>
      <th>ContentDownloadsPerMonth</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>WatchlistSize</th>
      <th>Churn</th>
      <th>...</th>
      <th>DeviceRegistered_Mobile</th>
      <th>DeviceRegistered_TV</th>
      <th>DeviceRegistered_Tablet</th>
      <th>GenrePreference_Comedy</th>
      <th>GenrePreference_Drama</th>
      <th>GenrePreference_Fantasy</th>
      <th>GenrePreference_Sci-Fi</th>
      <th>Gender_Male</th>
      <th>ParentalControl_Yes</th>
      <th>SubtitlesEnabled_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.169131</td>
      <td>-0.331703</td>
      <td>-1.012550</td>
      <td>1.445777</td>
      <td>-0.568906</td>
      <td>-1.005712</td>
      <td>-0.715179</td>
      <td>-0.175519</td>
      <td>-1.253786</td>
      <td>0</td>
      <td>...</td>
      <td>1.732672</td>
      <td>-0.575053</td>
      <td>-0.578590</td>
      <td>-0.501939</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>2.010816</td>
      <td>1.000299</td>
      <td>-1.001572</td>
      <td>-1.002353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.089945</td>
      <td>-1.690423</td>
      <td>-0.871303</td>
      <td>1.062671</td>
      <td>-1.317459</td>
      <td>-0.450971</td>
      <td>0.411960</td>
      <td>1.216976</td>
      <td>1.526687</td>
      <td>0</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>-0.575053</td>
      <td>1.728341</td>
      <td>-0.501939</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>1.000299</td>
      <td>-1.001572</td>
      <td>0.997652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.376731</td>
      <td>-0.088741</td>
      <td>0.254353</td>
      <td>-1.165718</td>
      <td>-0.691019</td>
      <td>-0.104258</td>
      <td>1.069988</td>
      <td>0.520728</td>
      <td>-1.531833</td>
      <td>0</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>-0.575053</td>
      <td>-0.578590</td>
      <td>-0.501939</td>
      <td>-0.499914</td>
      <td>1.994948</td>
      <td>-0.497310</td>
      <td>1.000299</td>
      <td>0.998430</td>
      <td>0.997652</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.819125</td>
      <td>-1.207816</td>
      <td>-0.990879</td>
      <td>0.663322</td>
      <td>0.777613</td>
      <td>0.381141</td>
      <td>1.102179</td>
      <td>-0.871766</td>
      <td>1.665711</td>
      <td>0</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>-0.575053</td>
      <td>1.728341</td>
      <td>-0.501939</td>
      <td>2.000344</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>1.000299</td>
      <td>0.998430</td>
      <td>0.997652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.089945</td>
      <td>1.031143</td>
      <td>0.412150</td>
      <td>-0.037246</td>
      <td>-0.928765</td>
      <td>-0.312285</td>
      <td>0.531014</td>
      <td>-0.175519</td>
      <td>-1.670857</td>
      <td>0</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>1.738970</td>
      <td>-0.578590</td>
      <td>1.992275</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>-0.999701</td>
      <td>-1.001572</td>
      <td>-1.002353</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
Now, let\'s perform the same transformations (one-hot encoding and
scaling) on the test data. We use the parameters (i.e., the category
mapping for one-hot encoding and the mean and standard deviation for
scaling) that we learned from the training data to transform the test
data. This is to ensure that the same transformation is applied
consistently across the training and test datasets.
:::

::: {.cell .code execution_count="13"}
``` python
# One-hot encode the categorical variables in the test data
test_encoded = ohe.transform(test_df[categorical_columns])
test_encoded = pd.DataFrame(test_encoded, columns=feature_names)

# Concatenate the original numerical columns and the one-hot encoded columns
test_data_encoded = pd.concat([test_df.select_dtypes(include=['int64', 'float64']), test_encoded], axis=1)

# Scale the numerical variables in the test data
test_data_encoded[numerical_columns] = scaler.transform(test_data_encoded[numerical_columns])

# Show the first few rows of the preprocessed test data
test_data_encoded.head()
```

::: {.output .execute_result execution_count="13"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>ViewingHoursPerWeek</th>
      <th>AverageViewingDuration</th>
      <th>ContentDownloadsPerMonth</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>WatchlistSize</th>
      <th>SubscriptionType_Premium</th>
      <th>...</th>
      <th>DeviceRegistered_Mobile</th>
      <th>DeviceRegistered_TV</th>
      <th>DeviceRegistered_Tablet</th>
      <th>GenrePreference_Comedy</th>
      <th>GenrePreference_Drama</th>
      <th>GenrePreference_Fantasy</th>
      <th>GenrePreference_Sci-Fi</th>
      <th>Gender_Male</th>
      <th>ParentalControl_Yes</th>
      <th>SubtitlesEnabled_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.644122</td>
      <td>1.242876</td>
      <td>-0.137084</td>
      <td>0.767017</td>
      <td>0.594196</td>
      <td>1.213252</td>
      <td>0.450126</td>
      <td>-0.871766</td>
      <td>1.526687</td>
      <td>1.420046</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>1.738970</td>
      <td>-0.578590</td>
      <td>1.992275</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>1.000299</td>
      <td>-1.001572</td>
      <td>-1.002353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.493400</td>
      <td>-0.595673</td>
      <td>0.023990</td>
      <td>1.456060</td>
      <td>-0.696379</td>
      <td>1.282595</td>
      <td>-0.849307</td>
      <td>-0.871766</td>
      <td>1.387664</td>
      <td>-0.704202</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>1.738970</td>
      <td>-0.578590</td>
      <td>-0.501939</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>-0.999701</td>
      <td>0.998430</td>
      <td>-1.002353</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.606640</td>
      <td>0.584230</td>
      <td>-1.291688</td>
      <td>-1.147346</td>
      <td>0.953367</td>
      <td>-0.728341</td>
      <td>1.561050</td>
      <td>-0.871766</td>
      <td>1.387664</td>
      <td>-0.704202</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>-0.575053</td>
      <td>-0.578590</td>
      <td>-0.501939</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>2.010816</td>
      <td>-0.999701</td>
      <td>-1.001572</td>
      <td>0.997652</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.814239</td>
      <td>0.662424</td>
      <td>1.148428</td>
      <td>1.341570</td>
      <td>1.677817</td>
      <td>-0.728341</td>
      <td>1.680308</td>
      <td>-1.568014</td>
      <td>1.526687</td>
      <td>-0.704202</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>-0.575053</td>
      <td>1.728341</td>
      <td>1.992275</td>
      <td>-0.499914</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>-0.999701</td>
      <td>0.998430</td>
      <td>0.997652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.901741</td>
      <td>-0.019563</td>
      <td>0.723051</td>
      <td>0.266946</td>
      <td>-0.434722</td>
      <td>-1.283082</td>
      <td>-0.134890</td>
      <td>0.520728</td>
      <td>-1.670857</td>
      <td>-0.704202</td>
      <td>...</td>
      <td>-0.577143</td>
      <td>-0.575053</td>
      <td>1.728341</td>
      <td>-0.501939</td>
      <td>2.000344</td>
      <td>-0.501266</td>
      <td>-0.497310</td>
      <td>-0.999701</td>
      <td>-1.001572</td>
      <td>-1.002353</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
# Logistic Regression
:::

::: {.cell .code execution_count="14"}
``` python
# Separate features and target from the training data
X_train = train_data_encoded.drop('Churn', axis=1)
y_train = train_data_encoded['Churn']

# Initialize the logistic regression model with balanced class weights
logreg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit the model on the resampled training data
logreg.fit(X_train, y_train)

# Predict the target for the training data
y_train_pred = logreg.predict(X_train)

# Print the classification report for the training data
print(classification_report(y_train, y_train_pred))

# Plot the confusion matrix for the training data
conf_mat = confusion_matrix(y_train, y_train_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for the Training Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.91      0.68      0.78    199605
               1       0.32      0.69      0.44     44182

        accuracy                           0.68    243787
       macro avg       0.61      0.68      0.61    243787
    weighted avg       0.80      0.68      0.71    243787
:::

::: {.output .display_data}
![](vertopal_bad73b54b05e4bdcb74fd894d0d06097/cad508a102f80dc7acbcefa34f2f2a7e0ce05731.png)
:::
:::

::: {.cell .markdown}
From the classification report, we can see that the logistic regression
model has a recall of 0.69 for the positive class (Churn = 1). Recall is
a measure of a model\'s ability to find all the relevant cases within a
dataset. The high recall indicates that the model is good at catching
positive instances (churned customers), which is our primary goal in
this context. This is good news because we want to identify as many
customers at risk of churning as possible so that we can intervene and
try to retain them.

However, the precision for the positive class (Churn = 1) is relatively
low (0.32), which means that among all the instances that the model
predicted as positive, only about 32% of them are actually positive. In
other words, the model has a high false positive rate. This is not
ideal, but it\'s a trade-off we make to achieve a high recall.

The confusion matrix further illustrates these points. The model has a
relatively high number of true positives (TP = 30373) and true negatives
(TN = 135843), but it also has a high number of false positives (FP =
63762) and a low number of false negatives (FN = 13809).

Since we\'re interested in identifying customers who are likely to
churn, having a higher recall (even at the cost of precision) can be
considered acceptable in this case. The company can then take action
(such as offering discounts or improved services) to try to retain these
customers.
:::

::: {.cell .code execution_count="15" scrolled="false"}
``` python
### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# Use our dummy classifier to make predictions on test_df using `predict_proba` method:
#predicted_probability = dummy_clf.predict_proba(test_df.drop(['CustomerID'], axis=1))[:, 1]
# Predict the target for the test data
predicted_probability = logreg.predict_proba(test_data_encoded)[:, 1]

# Create the prediction dataframe
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'],
    'predicted_probability': predicted_probability
})

print(prediction_df.shape)
prediction_df.head(10)
```

::: {.output .stream .stdout}
    (104480, 2)
:::

::: {.output .execute_result execution_count="15"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>predicted_probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1W6BHP6RM</td>
      <td>0.348485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LFR4X92X8H</td>
      <td>0.157539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>QM5GBIYODA</td>
      <td>0.758719</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D9RXTK2K9F</td>
      <td>0.172345</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENTCCHR1LR</td>
      <td>0.411773</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7A88BB5IO6</td>
      <td>0.799602</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70OMW9XEWR</td>
      <td>0.383766</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EL1RMFMPYL</td>
      <td>0.650321</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4IA2QPT6ZK</td>
      <td>0.534780</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AEDCWHSJDN</td>
      <td>0.499526</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
**PLEASE CHANGE CODE ABOVE TO IMPLEMENT YOUR OWN PREDICTIONS**
:::

::: {.cell .markdown}
## Final Tests - **IMPORTANT** - the cells below must be run prior to submission

Below are some tests to ensure your submission is in the correct format
for autograding. The autograding process accepts a csv
`prediction_submission.csv` which we will generate from our
`prediction_df` below. Please run the tests below an ensure no assertion
errors are thrown.
:::

::: {.cell .code execution_count="16"}
``` python
# FINAL TEST CELLS - please make sure all of your code is above these test cells

# Writing to csv for autograding purposes
prediction_df.to_csv("prediction_submission.csv", index=False)
submission = pd.read_csv("prediction_submission.csv")

assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_df.'
```
:::

::: {.cell .code execution_count="17"}
``` python
# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.columns[0] == 'CustomerID', 'The first column name should be CustomerID.'
assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'
```
:::

::: {.cell .code execution_count="18"}
``` python
# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.shape[0] == 104480, 'The dataframe prediction_df should have 104480 rows.'
```
:::

::: {.cell .code execution_count="19"}
``` python
# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'
```
:::

::: {.cell .code execution_count="20"}
``` python
# FINAL TEST CELLS - please make sure all of your code is above these test cells

## This cell calculates the auc score and is hidden. Submit Assignment to see AUC score.
```
:::

::: {.cell .markdown}
## SUBMIT YOUR WORK!

Once we are happy with our `prediction_df` and
`prediction_submission.csv` we can now submit for autograding! Submit by
using the blue **Submit Assignment** at the top of your notebook. Don\'t
worry if your initial submission isn\'t perfect as you have multiple
submission attempts and will obtain some feedback after each submission!
:::
