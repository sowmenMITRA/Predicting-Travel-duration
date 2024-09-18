# **Predictive Analysis of Transportation Data**

## **Introduction**
This project aims to leverage machine learning techniques to analyze transportation data. The primary goal is to build predictive models that can forecast outcomes based on historical data, such as predicting travel durations or estimating the likelihood of certain events based on environmental factors like weather conditions.

## **Objectives**
1. **Predictive Accuracy**: Develop models that provide accurate predictions of travel times and other transportation-related metrics.
2. **Insight Generation**: Extract meaningful insights from the data that can help in decision-making processes for urban planning and transportation management.
3. **Feature Engineering**: Identify and engineer relevant features from the dataset that significantly impact the predictive models.

## **Dataset Description**
The dataset contains over 9.6 million records and 26 variables related to transportation, including:

1. **Geographic Coordinates**: Start (PLong, PLatd) and destination (DLong, DLatd) positions.
2. **Temporal Data**: Month (Pmonth, Dmonth), day (Pday, Dday), hour (Phour, Dhour), and minutes (Pmin, Dmin) of travel.
3. **Environmental Factors**: Temperature (Temp), precipitation (Precip), wind (Wind), humidity (Humid), solar radiation (Solar), snowfall (Snow), ground temperature (GroundTemp), and dust particles (Dust).
4. **Transport Metrics**: Distance and duration of travels.

## **Methodology**
1. **Data Preprocessing**: Cleansing and transformation of raw data to prepare it for analysis.
2. **Exploratory Data Analysis (EDA)**: Initial exploration to understand the distributions, correlations, and basic characteristics of the data.
3. **Feature Engineering**: Creation of new features and modification of existing ones to enhance model performance.
4. **Model Development**: Utilization of various machine learning algorithms including Linear Regression, Random Forest, and LightGBM to develop predictive models.
5. **Model Evaluation**: Using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and the R-squared value to evaluate and compare the performance of different models.

## **Tools and Technologies**
1. **Python**: Primary programming language.
2. **Libraries**: `pandas`, `numpy` for data manipulation; `matplotlib`, `seaborn` for visualization; `scikit-learn`, `LightGBM` for machine learning.
## **Data Preprocessing**

### **1. Data Cleaning**
- **Handling Missing Values**:  
  Checked for any missing values in the dataset using `df.isnull().sum()`. Given the large size of the dataset, rows with missing values in critical columns (e.g., Distance, Duration) were removed, while minor missing entries in less impactful columns were imputed using appropriate strategies such as the median or mode.
  
- **Removing Duplicates**:  
  Removed duplicate entries to prevent data redundancy, which could skew the results of the analysis. This was done using `df.drop_duplicates()`.

### **2. Data Transformation**
- **Feature Encoding**:  
  Converted categorical variables into numerical format to facilitate their use in machine learning models. Used one-hot encoding for nominal variables with no intrinsic ordering (e.g., Day of the Week) through `pd.get_dummies()`. For ordinal variables, label encoding was applied where appropriate.

- **Feature Scaling**:  
  Standardized numerical features to have a mean of zero and a standard deviation of one. This normalization was crucial for models sensitive to the scale of input data, like k-nearest neighbors and gradient descent-based algorithms. Used `StandardScaler` from `scikit-learn`.

- **Date and Time Conversion**:  
  Extracted useful components from date and time columns, such as hour, day, month, and year, and transformed them into separate features to capture potential seasonal trends and daily patterns.

### **3. Feature Engineering**
- **New Features**:  
  Created new features that could potentially enhance model performance. For example, calculated the speed by dividing the distance by the duration for each trip. Also, engineered a feature representing the time of day as morning, afternoon, evening, or night to capture varying traffic conditions.

- **Dimensionality Reduction**:  
  Applied techniques such as Principal Component Analysis (PCA) to reduce the dimensionality of the dataset, focusing on retaining features that explain the most variance in the data.

### **4. Data Segmentation**
- **Splitting the Data**:  
  Divided the dataset into training and testing sets to evaluate the performance of machine learning models. Typically, the data was split into 80% for training and 20% for testing using the `train_test_split` method from `scikit-learn`, ensuring a random shuffle before the split.

### **5. Tools and Techniques**
- **Python Libraries**:  
  Utilized Python libraries such as `pandas` for data manipulation, `numpy` for numerical calculations, and `scikit-learn` for preprocessing and machine learning tasks.

- **Data Integrity Checks**:  
  Regular integrity checks were performed after each preprocessing step to ensure the accuracy and consistency of the data transformation processes.
  ## **Feature Engineering**

### **1. Creating Interaction Features**
- **Speed Calculation**:  
  Introduced a new feature, 'Speed', calculated by dividing 'Distance' by 'Duration' for each trip. This feature helps capture the velocity aspect of the trips, which may correlate with factors like time of day or weather conditions.
  
- **Time of Day**:  
  Developed a categorical variable 'Time_of_Day' from the 'Phour' feature, categorizing each trip into 'Morning', 'Afternoon', 'Evening', and 'Night'. This classification aims to capture traffic pattern variations throughout the day.

### **2. Transforming Temporal Data**
- **Weekday/Weekend**:  
  Created a binary feature 'Is_Weekend' indicating whether a trip occurred on a weekend. This feature is derived from 'PDweek' and 'DDweek' (day of the week), assuming different traffic patterns may be observed during weekends versus weekdays.

- **Seasonal Features**:  
  Engineered 'Season' based on 'Pmonth' to capture seasonal variations in transportation patterns. The months were categorized into 'Spring', 'Summer', 'Autumn', and 'Winter'.

### **3. Weather-Related Features**
- **Weather Index**:  
  Computed a 'Weather_Index' using a weighted average of weather-related variables such as 'Temp', 'Precip', 'Wind', and 'Humid'. This index provides a composite measure of weather conditions, potentially impacting travel times and distances.

- **Categorical Weather Conditions**:  
  Segmented continuous weather data like temperature and precipitation into categorical bins to simplify the model's interpretation of weather impacts on transportation metrics.

## **Model: LightGBM**
**LightGBM**, short for Light Gradient Boosting Machine, is a highly efficient gradient boosting framework designed for speed and performance. It's part of the Microsoft Distributed Machine Learning Toolkit and is popular for its ability to handle large-scale data efficiently, with a focus on distributed and efficient learning.

