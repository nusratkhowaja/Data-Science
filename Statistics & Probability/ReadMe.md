# Data-Driven Brand Strategy for a Cycle Share Scheme

## Objective

The objective of this project is to determine the brand persona for a new cycle sharing scheme in Seattle. By understanding the customer base, we aim to develop targeted marketing strategies to improve customer retention and attract new users.

## Scenario

The cycle sharing scheme offers a convenient, affordable, and environmentally friendly transportation option within the city. Currently, it operates with 500 bikes at 50 stations. Users can choose between an annual membership offering quick access and unlimited 45-minute rentals, or short-term passes granting access for 24 hours or 3 days. Bikes can be picked up and returned at any station citywide.

## Assumptions

1. **Assumption:** Short-term pass holders are likely to be new users testing the service.

2. **Assumption:** Millennial customers are more likely to be loyal to brands they like.

3. **Assumption:** Trip duration and frequency may vary based on factors such as user demographics, subscription type, and station location.

4. **Assumption:** Understanding the seasonality of trip frequency and duration can help optimize operational and marketing strategies.

5. **Assumption:** Outliers in trip duration data may skew the mean and impact analysis.

6. **Assumption:** There may be correlations between certain user characteristics (e.g., age, gender) and trip behavior.


## Data and Analysis

We utilized a dataset containing transaction history information from the cycle sharing scheme. The analysis included univariate and multivariate analyses, correlation analysis, and time series analysis to understand user behavior and demographics.


## Feature Dictionary for Cycle Sharing Scheme Analysis

| **Feature Name** | **Description** | **Data Type** |
|---|---|---|
| trip_id | Unique ID assigned to each trip | String |
| starttime | Day and time when the trip started, in PST | String (e.g., "2023-10-26 15:00:00") |
| stoptime | Day and time when the trip ended, in PST | String (e.g., "2023-10-26 15:30:00") |
| bikeid | ID attached to each bike | String |
| tripduration | Time of trip in seconds | Integer |
| from_station_name | Name of station where the trip originated | String |
| to_station_name | Name of station where the trip terminated | String |
| from_station_id | ID of station where trip originated | String or Integer |
| to_station_id | ID of station where trip terminated | String or Integer |
| usertype | Value can include either: "short-term pass holder" or "member" | Categorical |
| gender | Gender of the rider | Categorical |
| birthyear | Birth year of the rider | Integer |




## Preliminary Analysis


```python
# importing the packages needed

import random
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import scipy
from scipy import stats
import seaborn
```


```python
# reading the csv file
data = pd.read_csv("cycle_trips_dataset.csv")
```

As a first step, determining the size of the dataset and seeing the top 5 observations helps us to get a glimpse of how our data looks


```python
print(len(data))         # len is used to check the length
data.head(5)            # head fucntion is used to view the topmost observations of the dataset
```

    236065





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>starttime</th>
      <th>stoptime</th>
      <th>bikeid</th>
      <th>tripduration</th>
      <th>from_station_name</th>
      <th>to_station_name</th>
      <th>from_station_id</th>
      <th>to_station_id</th>
      <th>usertype</th>
      <th>gender</th>
      <th>birthyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>431</td>
      <td>10/13/2014 10:31</td>
      <td>10/13/2014 10:48</td>
      <td>SEA00298</td>
      <td>985.935</td>
      <td>2nd Ave &amp; Spring St</td>
      <td>Occidental Park / Occidental Ave S &amp; S Washing...</td>
      <td>CBD-06</td>
      <td>PS-04</td>
      <td>Member</td>
      <td>Male</td>
      <td>1960.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>432</td>
      <td>10/13/2014 10:32</td>
      <td>10/13/2014 10:48</td>
      <td>SEA00195</td>
      <td>926.375</td>
      <td>2nd Ave &amp; Spring St</td>
      <td>Occidental Park / Occidental Ave S &amp; S Washing...</td>
      <td>CBD-06</td>
      <td>PS-04</td>
      <td>Member</td>
      <td>Male</td>
      <td>1970.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>433</td>
      <td>10/13/2014 10:33</td>
      <td>10/13/2014 10:48</td>
      <td>SEA00486</td>
      <td>883.831</td>
      <td>2nd Ave &amp; Spring St</td>
      <td>Occidental Park / Occidental Ave S &amp; S Washing...</td>
      <td>CBD-06</td>
      <td>PS-04</td>
      <td>Member</td>
      <td>Female</td>
      <td>1988.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>434</td>
      <td>10/13/2014 10:34</td>
      <td>10/13/2014 10:48</td>
      <td>SEA00333</td>
      <td>865.937</td>
      <td>2nd Ave &amp; Spring St</td>
      <td>Occidental Park / Occidental Ave S &amp; S Washing...</td>
      <td>CBD-06</td>
      <td>PS-04</td>
      <td>Member</td>
      <td>Female</td>
      <td>1977.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>435</td>
      <td>10/13/2014 10:34</td>
      <td>10/13/2014 10:49</td>
      <td>SEA00202</td>
      <td>923.923</td>
      <td>2nd Ave &amp; Spring St</td>
      <td>Occidental Park / Occidental Ave S &amp; S Washing...</td>
      <td>CBD-06</td>
      <td>PS-04</td>
      <td>Member</td>
      <td>Male</td>
      <td>1971.0</td>
    </tr>
  </tbody>
</table>
</div>



We need to determine the type of each column whether it is an integer, string, etc. We will use the dtypes command


```python
data.dtypes
```




    trip_id                int64
    starttime             object
    stoptime              object
    bikeid                object
    tripduration         float64
    from_station_name     object
    to_station_name       object
    from_station_id       object
    to_station_id         object
    usertype              object
    gender                object
    birthyear            float64
    dtype: object



##  Univariate analysis

This is the analysis performed on a single variable and thus does not account for any sort of relationship among exploratory variables. We perform univariate analysis on the dataset to better understand the features in isolation

we want to print the date range that starts from the first value of starttime and ends with the last value of stoptime


```python
data = data.sort_values(by='starttime')
data.reset_index()
print ('Date range of dataset: %s - %s' % (data.loc[1, 'starttime'], data.loc[len(data)-1, 'stoptime']))
```

    Date range of dataset: 10/13/2014 10:32 - 9/1/2016 0:20


We can draw two important insights from above:
    
    1. the data ranges from October 2014 up till September 2016 (i.e., three years of data)
    2. it seems like the cycle sharing service is usually operational beyond the standard 9 to 5 business hours.

### Type of Memberships

The next step is to determine whether more people prefer buying short term pass or long term rentals.

To determine that we need to plot a bar graph of trip frequencies by user type


```python
groupby_user = data.groupby('usertype').size()
groupby_user.plot.bar(title = 'Distribution of User by membership')
```




    <AxesSubplot:title={'center':'Distribution of User by membership'}, xlabel='usertype'>




    
![data_head.png](/persona_files/persona_16_1.png)
    


Conclusion : Users prefer memberships more than the Short-term pass. It must be the case that new users would be short-term pass holders however once they try out the service and
become satisfied would ultimately avail the membership to receive the perks and benefits
offered.

### Distribution By gender

The next task is to determine whether more males or females use the cycle service or they both equally utilize it


```python
groupby_gender = data.groupby('gender').size()
groupby_gender.plot.bar(title = 'Distribution of Trips by Gender')
```




    <AxesSubplot:title={'center':'Distribution of Trips by Gender'}, xlabel='gender'>




    
![data_head.png](/persona_files/persona_19_1.png)
    


Conclusion : In conclusion, we can say that the number of Males that utilize the cycle service is more than three times that of the number of females.

### Target Age Group

We need to know more about the target customers to whom to company’s marketing message will be targetted to. For that we can create an age-wise distribution of our data in order to know the demographics in a better way


```python
data = data.sort_values(by = 'birthyear')
groupby_birthyear = data.groupby('birthyear').size()
groupby_birthyear.plot.bar(title = 'Distribution by Birth Year', figsize = (15,4))

```




    <AxesSubplot:title={'center':'Distribution by Birth Year'}, xlabel='birthyear'>




    
![data_head.png](/persona_files/persona_22_1.png)
    


Conclusion : We can see that most of the people belong to the birthyear 1980 - 1990 i.e. they belonged to the Gen Y (also known as millennials). Recent reports published by Elite Daily and CrowdTwist which said that millennials are the most loyal generation Hto their favorite brands. 

hence we can say that most of the millennials would be members rather than being short-term pass holders. In order to check this notion, look at the following code.


```python
data_mil = data[(data['birthyear'] >= 1977) & (data['birthyear'] <= 1994)]
groupby_mil = data_mil.groupby('usertype').size()
groupby_mil.plot.bar(title = 'Distibution of millenial user type')
```




    <AxesSubplot:title={'center':'Distibution of millenial user type'}, xlabel='usertype'>




    
![data_head.png](/persona_files/persona_24_1.png)
    


Conclusion : The notion stated by the reports appear to be true as all the millennials are members rather than being short-term card holders

## Multivariate Analysis 

Multivariate analysis refers to incorporation of multiple exploratory variables to
understand the behavior of a response variable. This seems to be the most feasible
and realistic approach considering the fact that entities within this world are usually
interconnected. Thus the variability in response variable might be affected by the
variability in the interconnected exploratory variables.



We want to validate if we have the field birthyear only for the "members" and not for "short term card holders"


```python
data[data['usertype'] == 'Short-Term Pass Holder']['birthyear'].isnull().values.all()
```




    True



So we see above that the birthyear is given only for the members. So the short term pass holders are not required to provide their birth years which is an inconsistency as this made our prior conclusion " Millennials are very loyal to the brands they like" incorrect.

Now we validate the same thing for gender


```python
data[data['usertype'] == 'Short-Term Pass Holder']['gender'].isnull().values.all()
```




    True



From the output above we can conclude that we do not have any demographic details for short term pass holders. 

#### Time series Analysis

We are interested to see as to how the frequency of trips vary across date and time. But in order to make a time series plot, we need to convert the date from string format to date-time format and thereafter we need to split the date-time into date components (year, month, day, hour, etc.)


```python
List_ = list(data['starttime'])
List_ = [datetime.datetime.strptime(x, "%m/%d/%Y %H:%M") for x in List_]
data['starttime_mod'] = pd.Series(List_,index=data.index)
data['starttime_date'] = pd.Series([x.date() for x in List_],index=data.index)
data['starttime_year'] = pd.Series([x.year for x in List_],index=data.index)
data['starttime_month'] = pd.Series([x.month for x in List_],index=data.index)
data['starttime_day'] = pd.Series([x.day for x in List_],index=data.index)
data['starttime_hour'] = pd.Series([x.hour for x in List_],index=data.index)
```


```python
data.groupby('starttime_date')['tripduration'].mean().plot.bar(title = 'Distribution of Trip duration by date', figsize = (15,4))
```




    <AxesSubplot:title={'center':'Distribution of Trip duration by date'}, xlabel='starttime_date'>




    
![data_head.png](/persona_files/persona_36_1.png)
    


There seems to exist a pattern in the above time series analysis

the pattern is repeating over a fixed interval of time— that is, seasonality. In fact, we can split the distribution into three distributions. One pattern is the seasonality that is repeating over time. The second one is a flat density distribution. Finally, the last pattern is the lines (that is, the hikes) over that density function. In case of time series prediction we can make estimations for a future time using both of these distributions and add up in order to predict upon a calculated confidence interval.

#### Distribution of trips by year


```python
data.groupby('starttime_year')['tripduration'].mean().plot.bar(title = 'Distribution of Trip duration by year', figsize = (15,4))
```




    <AxesSubplot:title={'center':'Distribution of Trip duration by year'}, xlabel='starttime_year'>




    
![data_head.png](/persona_files/persona_39_1.png)
    


In the above, we can see a trend that the mean duration of trips is increasing on a yearly basis

#### Distribution of Trip by Months


```python
data.groupby('starttime_month')['tripduration'].mean().plot.bar(title = 'Distribution of Trip duration by date', figsize = (15,4))
```




    <AxesSubplot:title={'center':'Distribution of Trip duration by date'}, xlabel='starttime_month'>




    
![data_head.png](/persona_files/persona_42_1.png)
    


We can conclude that the duration increases first, then reaches its peak in the month of JULY and then starts to decline. The highest duration of trips is in the month of July

#### Distribution of trips by day


```python
data.groupby('starttime_day')['tripduration'].mean().plot.bar(title = 'Distribution of Trip duration by date', figsize = (15,4))
```




    <AxesSubplot:title={'center':'Distribution of Trip duration by date'}, xlabel='starttime_day'>




    
![data_head.png](/persona_files/persona_45_1.png)
    


We can conclude that there isn't any evident pattern in the trip duration by day; it is quite fluctuating

### Measuring Center of Measure

measures like mean, median, and mode help give a summary view of the features in question.

#### Trip duration Analysis

We need to know the mean and median of each trip.
Also we are interested in finding the station from which most of the trips originate, so as to run promotional schemes for the existing customers over there. For this, we will utilize the mode feature of the statistics package


```python
from collections import Counter
trip_duration = list(data['tripduration'])
Station_from = list(data['from_station_name'])
data_mode = Counter(Station_from)
print('The mean duration of trip is : %f' %statistics.mean(trip_duration))
print('The median duration of trip is : %f' %statistics.median(trip_duration))
print('The Station from which most trips originate is : %s' %data_mode.most_common(1))
mean_trip_duration = statistics.mean(trip_duration)
```

    The mean duration of trip is : 1202.612210
    The median duration of trip is : 633.235000
    The Station from which most trips originate is : [('Pier 69 / Alaskan Way & Clay St', 11274)]


Conclusion: Most of the trips originate from Pier 69 and Alaskan Way & Clay St. Hence this is the ideal location for running promotional campaigns targeted to existing customers
We can also see that the value of mean is quite higher than the median. Hence there might be some outliers present. We need to plot the distribution of the tripduration in order to explore this point.    


```python
data['tripduration'].plot.hist(bins=100, title='Frequency distribution of Trip duration')
plt.show()
```


    
![data_head.png](/persona_files/persona_52_0.png)
    


Conclusion: The extreme values on the right side are not very frequent but their extreme nature tends to increase the value of the mean

#### Box plot to determine outliers


```python
box = data.boxplot(column=['tripduration'])
plt.show()
```


    
![data_head.png](/persona_files/persona_55_0.png)
    


Conclusion : There are a huge number of outliers in the tripduration feature. We need to now determine the proportion of outliers to understand whether they are in majority or minority


```python
q75, q25 = np.percentile(trip_duration, [75 ,25])
iqr = q75 - q25
Percent_outlier = ((len(data) - len([x for x in trip_duration if q75+(1.5*iqr) >= x >= q25-(1.5*iqr)]))*100/float(len(data)))
print ('Proportion of values as outlier: %f percent' %Percent_outlier)
```

    Proportion of values as outlier: 9.548218 percent


Conclusion : As the data is time series data, we cannot remove the outliers. The best thing to do is apply some kind of transformation.
    In order to do that, we need to first find the mean of all the non outliers


```python
List_non_outlier = list(x for x in trip_duration if q75+(1.5*iqr) >= x >= q25-(1.5*iqr))
mean_non_outlier = statistics.mean(List_non_outlier)
print('The mean of the non outliers is %f '%mean_non_outlier)
```

    The mean of the non outliers is 711.726573 


Conclusion: The mean of non-outlier trip duration values  (i.e., approximately 712) is considerably lower than that calculated in the presence of outliers  (i.e., approximately 1,203). This best describes the notion that mean is highly affected by the
presence of outliers in the dataset.

#### Function to transform outliers


```python
upper_whisker = q75 + (1.5*iqr)

def transform_tripduration(x):
    if x > upper_whisker:
        return mean_trip_duration
    return x

data['tripduration_mean'] = data['tripduration'].apply(lambda x: transform_tripduration(x))
data['tripduration_mean'].plot.hist(bins=100, title='Frequency Distribution of mean transformed trip duration')
plt.show()

print ('Mean of trip duration: %f'%data['tripduration_mean'].mean())
print ('Standard deviation of trip duration: %f'%data['tripduration_mean'].std())
print ('Median of trip duration: %f'%data['tripduration_mean'].median())
```


    
![data_head.png](/persona_files/persona_62_0.png)
    


    Mean of trip duration: 758.597403
    Standard deviation of trip duration: 458.788345
    Median of trip duration: 633.235000


#### Finding the tripduration centre of measures for MALES


```python

```


```python
data_males =  data[(data['gender'] == 'Male')]
trip_duration_male = list(data_males['tripduration'])
Station_from_male = list(data_males['from_station_name'])
data_mode_males = Counter(Station_from_male)
print('The mean duration of trip is : %f' %statistics.mean(trip_duration_male))
print('The median duration of trip is : %f' %statistics.median(trip_duration_male))
print('The Station from which most trips originate is : %s' %data_mode_males.most_common(1))
mean_trip_duration_male = statistics.mean(trip_duration_male)
```

    The mean duration of trip is : 563.402797
    The median duration of trip is : 458.451500
    The Station from which most trips originate is : [('E Pine St & 16th Ave', 5888)]



```python
data_males['tripduration'].plot.hist(bins=100, title='Frequency distribution of Trip duration of males')
plt.show()
```


    
![data_head.png](/persona_files/persona_66_0.png)
    


Conclusion: The distribution seems to be slightly positively skewed. So we need to check for outliers using the box plot method


```python
box_males = data_males.boxplot(column=['tripduration'])
plt.show()
```


    
![data_head.png](/persona_files/persona_68_0.png)
    


Conclusion: There are a lot of outliers in male trips as well. Let us transform those outliers.


```python
q75, q25 = np.percentile(trip_duration_male, [75 ,25])
iqr = q75 - q25
upper_whisker_males = q75 + (1.5*iqr)

def transform_males(x):
    if x > upper_whisker_males:
        return mean_trip_duration_male
    return x

data_males['tripduration_mean'] = data_males['tripduration'].apply(lambda x: transform_males(x))
data_males['tripduration_mean'].plot.hist(bins=100, title='Frequency Distribution of mean transformed trip duration of males')
plt.show()

print ('Mean of trip duration of males: %f'%data_males['tripduration_mean'].mean())
print ('Standard deviation of trip duration of males: %f'%data_males['tripduration_mean'].std())
print ('Median of trip duration of males: %f'%data_males['tripduration_mean'].median())
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.



    
![data_head.png](/persona_files/persona_70_1.png)
    


    Mean of trip duration of males: 486.619138
    Standard deviation of trip duration of males: 227.516710
    Median of trip duration of males: 458.451500


## Correlation Analysis

### Determining the strength of relationships between variables

Correlation refers to the strength and direction of the relationship between two
quantitative features. A correlation value of 1 means strong correlation in the positive
direction, whereas a correlation value of -1 means a strong correlation in the negative
direction. A value of 0 means no correlation between the quantitative features.

An interesting question to ask is whether change in age brings a change in trip duration? Let's find out using correlation


```python
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
data['age'] = data['starttime_year'] - data['birthyear']
correlations = data[['tripduration','age']].corr(method='pearson')
print(correlations)
```

                  tripduration    age
    tripduration         1.000  0.058
    age                  0.058  1.000


the correlation came out to be weak and positive in nature hence there is no clear relation between these two

# Conclusions

Trip duration follows a definite seasonal pattern that
repeats over time. Forecasting this time series can help us predict the times when
the company needs to push its marketing efforts and times when most trips anticipated
can help ensure operational efficiencies.

As for the promotions, we now know that
the best station at which to kick off the campaign would be Pier 69/Alaskan Way & Clay St.


```python

```
