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

![1](https://github.com/user-attachments/assets/6e098d13-934f-4d81-bf7e-888f8b614a7c)

## **Model Development**
In this project, we explored various machine learning models to predict transportation-related metrics accurately. Our primary goal was to identify a model that not only provides precise predictions but also operates efficiently on large datasets. After rigorous testing and evaluation, the **LightGBM** model emerged as the most effective, outperforming both **Linear Regression** and **Random Forest** models in all key performance metrics.

### **Model Implementation and Evaluation**

#### **1. Linear Regression Model**
- **Configuration and Training**:  
  Implemented using a pipeline that included `StandardScaler` for feature normalization and `LinearRegression`. The model was trained on a 30% subset of the data to manage computational resources effectively.
  
- **Performance Metrics**:  
  - **Mean Absolute Error (MAE)**: 10.102  
  - **Mean Squared Error (MSE)**: 271.007  
  - **Root Mean Squared Error (RMSE)**: 16.462  
  - **R-squared**: 0.568  
  - **Cross-Validation MSE**: 269.704  

#### **2. Random Forest Model**
- **Configuration and Training**:  
  Configured with a simplified setup involving fewer trees and reduced depth to speed up training times, applied to a 10% sample of the data.
  
- **Performance Metrics**:  
  - **MAE**: 9.508  
  - **MSE**: 235.163  
  - **RMSE**: 15.335  
  - **R-squared**: 0.625  
  - **Cross-Validation MSE**: 234.602  

#### **3. LightGBM Model**
- **Configuration and Training**:  
  The model was tuned with optimal settings including `num_leaves`, `max_depth`, `learning_rate`, and `n_estimators`. It was trained using a consistent 10% sample of the data for an accurate comparison with other models.
  
- **Superior Performance Metrics**:  
  - **MAE**: 5.534  
  - **MSE**: 87.321  
  - **RMSE**: 9.345  
  - **R-squared**: 0.861  
  - **Cross-Validation MSE**: 88.116  

## **Comparative Analysis**

| **Model**           | **MAE** | **MSE**  | **RMSE** | **R-squared** |
|---------------------|---------|----------|----------|---------------|
| **Linear Regression** | 10.102  | 271.007  | 16.462   | 0.568         |
| **Random Forest**    | 9.508   | 235.163  | 15.335   | 0.625         |
| **LightGBM**         | 5.534   | 87.321   | 9.345    | 0.861         |

### **Accuracy and Efficiency**
- **LightGBM** not only achieved the lowest error metrics (MAE, MSE, RMSE) but also exhibited the highest **R-squared** value among the tested models, indicating its superior accuracy and ability to explain a significant proportion of the variance in the data.

### **Scalability**
Despite using the same fraction of the dataset for training as the **Random Forest** model, **LightGBM** demonstrated better scalability and efficiency, crucial for handling large-scale data in practical applications.

### **Robustness**
The low cross-validation **MSE** score reflects **LightGBM**'s robustness, showing minimal overfitting and strong generalization capabilities across different subsets of data.

### **Mean Absolute Error (MAE) of LightGBM**
The **LightGBM** model excelled in minimizing the **Mean Absolute Error**, registering a remarkable **MAE of 5.534**. This performance is significantly superior compared to the MAEs reported for **Linear Regression** (10.102) and **Random Forest** (9.508). The lower MAE value demonstrates **LightGBM**'s precision in forecasting, suggesting that its predictions are closely aligned with the actual values. This accuracy is crucial in applications where even small deviations can lead to substantial operational inefficiencies or financial losses.

![2](https://github.com/user-attachments/assets/1e46d804-b839-45fa-b4b4-b8a57dc914ae)

### **Mean Squared Error (MSE) of LightGBM**
**LightGBM** also led in reducing the **Mean Squared Error**, achieving an **MSE of 87.321**, which starkly contrasts with the higher MSEs of 271.007 for **Linear Regression** and 235.163 for **Random Forest**. The MSE metric, which squares the errors before averaging them, penalizes larger errors more than smaller ones, indicating that **LightGBM** effectively limits large prediction errors. This capability makes **LightGBM** particularly valuable in scenarios where outliers or large prediction errors could have disproportionate impacts.

![3](https://github.com/user-attachments/assets/9ca34d15-9ec0-40e5-aa2f-88520bd5ff15)

### **Root Mean Squared Error (RMSE) of LightGBM**
The **RMSE for LightGBM** was computed to be **9.345**, a testament to the model's consistent and reliable performance. This RMSE is significantly lower than those of the competing models, indicating not only the small variability in prediction errors but also a high level of consistency across different datasets or conditions. RMSE, being the square root of MSE, also makes it a more interpretable metric in terms of the original units of the data, providing a clear indication of the average magnitude of the prediction errors.

![4](https://github.com/user-attachments/assets/8ec845e2-f18a-4e98-bd14-fd621e628641)

### **R-squared Value of LightGBM**
The **R-squared value** achieved by **LightGBM**, standing at **0.861**, highlights its efficiency in explaining the variance within the dependent variable. This is considerably higher than the values noted for the other models, reflecting **LightGBM**'s ability to capture and explain the variability in the dataset effectively. A high R-squared value is indicative of a model that can not only predict accurately but also reflect the true dynamics of the underlying data, making it extremely effective for both predictive and explanatory purposes.

![5](https://github.com/user-attachments/assets/6067739e-2e69-43d0-aa87-051a01088ce4)

### **Cross-Validation MSE Comparison**
Cross-validation is crucial for assessing the generalization capabilities of a model beyond the specific data on which it was trained. In this project, we employed **K-Fold cross-validation** to evaluate and compare the robustness of the three models: **LightGBM**, **Linear Regression**, and **Random Forest**.

- **Linear Regression**: The cross-validation **MSE** for the Linear Regression model was **269.704**. This relatively high value suggests that while the model provides a decent fit to the data it has seen, its predictions vary more when confronted with new, unseen data, potentially indicating overfitting or an inability to capture all underlying patterns effectively.

- **Random Forest**: The Random Forest model yielded a cross-validation **MSE** of **234.602**, which is lower than that of Linear Regression but still indicates room for improvement. The reduced error in Random Forest compared to Linear Regression reflects its enhanced capability to handle non-linear relationships and interactions between features, thanks to its ensemble approach.

- **LightGBM**: **LightGBM** showed the best performance with a cross-validation **MSE** of **88.116**. This significantly lower MSE underscores **LightGBM**'s superior generalization ability across different subsets of data. The strength of **LightGBM** lies in its gradient boosting framework, which effectively reduces bias and variance, leading to more accurate and consistent predictions across varied datasets.

### **Analysis**
The stark contrast in cross-validation MSE values highlights the effectiveness of **LightGBM** in not just fitting the training data but also in maintaining accuracy and reliability when applied to new, unseen data. This robustness makes **LightGBM** particularly suitable for real-world applications where models frequently encounter data that differ from the conditions they were trained on. In contrast, both **Linear Regression** and **Random Forest**, while useful in their own right, display limitations in their generalization capabilities as evidenced by their higher MSE values in cross-validation.
## **Exhaustive Evaluation of the LightGBM Model**

The exhaustive evaluation of the **LightGBM** model involved a detailed analysis of feature importances, learning curves, and cross-validation performance to assess the model's effectiveness, reliability, and robustness. Below is a summary of each component:

### **1. Feature Importances**

The analysis of **feature importances** revealed which predictors most significantly influence the model's outcomes. Key features such as **"Distance," "Phour," "Dhour,"** and **"Dmin"** were identified as the top contributors to the model’s predictions. Understanding these important features helps prioritize data collection and preprocessing, which can further enhance model performance and efficiency.

### **2. Learning Curves**

**Learning curves** were generated to track how well the LightGBM model learns as the training data increases. The curves demonstrated a consistent decrease in negative MSE for training scores, indicating continuous model improvement. However, the **cross-validation scores plateaued**, suggesting that after a certain point, adding more data does not significantly improve generalization performance. This plateau reflects the model's optimal training size for balancing generalization and efficiency.

### **3. Cross-Validation Performance**

Cross-validation was used to ensure that the model's performance was not specific to a single train-test split. The **cross-validation MSE** offered a reliable estimate of LightGBM's generalization capability. The consistency of MSE across folds underscored the model’s ability to maintain accuracy and reliability, making it suitable for real-world applications where data variability is common.

### **Summary of Evaluation**

The evaluation highlights **LightGBM**'s strengths in:
- **Feature prioritization** for enhancing accuracy.
- Efficient learning with an optimal training size.
- Robust generalization across different subsets of data.

This exhaustive evaluation ensures that **LightGBM** is well-equipped for deployment in real-world scenarios, delivering both accuracy and consistency in predictions.

---

## **Conclusion: Justification for Choosing LightGBM as the Preferred Model**

After a comprehensive evaluation and comparison of multiple machine learning models, **LightGBM** was chosen as the optimal model for predicting key transportation metrics. This decision is supported by the following factors:

### **1. Superior Efficiency**
**LightGBM** employs a **leaf-wise growth algorithm**, in contrast to the traditional level-wise approach used in other gradient boosting models. This innovation enhances **training speed** and **loss minimization**, making **LightGBM** particularly adept at handling large datasets while being both time-efficient and resource-efficient.

### **2. High Predictive Accuracy**
**LightGBM** consistently achieved:
- **Lower Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** than other models like **Linear Regression** and **Random Forest**.
- This exceptional accuracy is critical for forecasting scenarios where **complex, non-linear relationships** between variables must be captured.

### **3. Robust Generalization**
The **cross-validation MSE** was significantly lower for LightGBM, confirming its ability to generalize well across different subsets of data. This is particularly important for ensuring consistent performance in real-world environments where new, unseen data can present challenges.

### **4. Feature Importance Insights**
The **feature importance analysis** revealed that **LightGBM** effectively prioritized influential predictors, such as **'Distance'** and **'Phour'**, improving both the precision and interpretability of the model’s forecasts.

![6](https://github.com/user-attachments/assets/ec802a68-b336-4591-874b-960477c14e7b)

### **5. Learning Curve Insights**
The generated **learning curves** showed that **LightGBM** efficiently utilizes increasing amounts of data without overfitting, making it scalable for larger datasets and maintaining sustained performance improvements as more data becomes available.

![7](https://github.com/user-attachments/assets/f86dc8e0-4d1e-42ba-a777-e99d4ac25f82)

---

### **Conclusion**

In summary, **LightGBM**'s blend of:
- **Scalability** and **efficiency**,
- **Precision** in handling complex datasets,
- And **robustness** in generalization

make it the definitive choice for this project. LightGBM not only meets but exceeds the requirements for deployment in operational settings, promising **reliable** and **actionable insights** from the data.

