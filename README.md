# Project Documentation

## Overview

This project aims to predict shipment delays based on various factors such as weather conditions, traffic conditions, vehicle type, and more. The dataset includes information about shipments, including the origin and destination cities, weather conditions, vehicle type, and shipment dates. The goal is to use machine learning models to predict whether a shipment will be delayed.

The following steps were carried out to preprocess the data, perform exploratory data analysis (EDA), train machine learning models, and evaluate their performance.

---

## Steps Taken

### 1. Data Loading and Inspection

The dataset was loaded from an Excel file. The first step was to inspect the data using the head and info functions to understand its structure and identify any missing or malformed data.

1. Purpose: Understand the basic structure of the dataset and check for missing values or duplicates.

### 2. Handling Missing Data

The column `Vehicle Type` had missing values. To address this issue, the missing values in the `Vehicle Type` column were filled with a placeholder value `Unknown`. This approach avoids losing rows by removing missing data, which could be valuable in model training.

1. Purpose: Ensure no loss of data and handle missing values appropriately.

### 3. Duplicate Data Removal

Next, I checked for any duplicate rows in the dataset using the duplicated function. This ensures that I am working with unique entries in the data, as duplicate entries could skew the analysis and model results.

1. Purpose: Remove duplicates to ensure the dataset is clean and does not contain redundant information.

### 4. Data Cleaning and Normalization

The column names were normalized by converting them to lowercase and replacing spaces with underscores. This ensures consistency and makes it easier to reference columns programmatically.

Additionally, the `shipment_id` column was dropped as it was irrelevant for modeling, and its removal reduces the dimensionality of the dataset.

1. Purpose: Improve readability and manageability of the dataset by standardizing column names.

### 5. Exploratory Data Analysis (EDA)

#### Analyzing Delayed Shipments by Distance

I explored the relationship between shipment distance and whether the shipment was delayed. By comparing the mean, median, and percentiles of distance for delayed and non-delayed shipments, I observed that distance alone may not be a significant factor contributing to delays.

1. Purpose: Investigate the relationship between distance and delays to determine if distance is a key predictor of delay.

#### Analyzing Delayed Shipments by Weather Conditions

I examined the distribution of delayed shipments across different weather conditions. The data revealed that fog, rain, and storm conditions always resulted in delayed shipments, indicating a near-100 percent chance of delay under these conditions.

1. Purpose: Identify weather conditions that could be strong predictors of shipment delays.

#### Analyzing Delayed Shipments by Traffic Conditions

I also analyzed the traffic conditions and their impact on delays. The analysis showed that heavy and moderate traffic are strongly linked to delays, while light traffic still results in a significant number of delayed shipments.

1. Purpose: Understand the influence of traffic conditions on shipment delays and identify key features.

#### Analyzing Delayed Shipments by Vehicle Type

All vehicle types exhibited more delayed shipments than non-delayed ones, indicating that vehicle type may play a role in predicting delays.

1. Purpose: Investigate the impact of vehicle type on shipment delays.

#### Analyzing Delayed Shipments by Origin and Destination

I grouped the data by origin and destination to analyze the distribution of delays across different cities. Although delays were common across all origin and destination cities, there was no substantial variation, suggesting that the origin and destination cities alone may not be significant predictors of delays.

1. Purpose: Examine whether the origin and destination cities have a significant impact on delays.

### 6. Data Visualization

Several visualizations were created to better understand the relationships between features and delays:

1. Boxplot: Used to visualize the distribution of shipment distances for delayed and non-delayed shipments.
2. Countplots: Created to visualize the distribution of delayed shipments across different weather conditions, traffic conditions, vehicle types, origins, and destinations.

These visualizations helped to further clarify the relationships between different features and the target variable shipment delay.

### 7. Data Encoding

Categorical variables such as `origin`, `destination`, `weather_conditions`, `traffic_conditions`, and `vehicle_type` were encoded using one-hot encoding to convert them into numerical format. This step ensures that the machine learning models can interpret these variables.

1. Purpose: Convert categorical data into a format that machine learning algorithms can process.

Additionally, the `delayed` column was mapped to binary values (1 for Yes and 0 for No) for use as the target variable in the model.

### 8. Feature Scaling

I applied feature scaling to the `shipment_date` and `planned_delivery_date` columns, converting them to a numerical format in seconds. These columns were then scaled using StandardScaler to normalize the data and improve model performance.

1. Purpose: Ensure that all features have the same scale, preventing features with larger numerical ranges from dominating the model.

### 9. Model Training

Three different machine learning models were trained and evaluated: Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.

1. Logistic Regression: Chosen for its simplicity and ability to perform well on binary classification problems.
2. Decision Tree Classifier: Selected for its interpretability and ability to model complex relationships.
3. Random Forest Classifier: Used for its robustness and ability to handle a large number of features.

The models were trained on the training data, and predictions were made on the test data.

### 10. Model Evaluation

Each model was evaluated using accuracy, precision, recall, and F1 score to determine the best-performing model. Based on the evaluation metrics, Logistic Regression performed the best overall, achieving the highest F1 score (0.9360) and perfect precision (1.0000) while maintaining a high recall (0.8796).

1. Purpose: Compare the performance of different models and select the best one based on evaluation metrics.

### 11. Model Saving

The best-performing models (Logistic Regression, Decision Tree, and Random Forest) were saved using joblib for future use. Additionally, the scaler used in Logistic Regression was also saved to ensure consistent preprocessing in the future.

---

## Conclusion

Based on the analysis and model evaluation, Logistic Regression was selected as the best model for predicting shipment delays. This model achieved high accuracy, precision, recall, and F1 score, making it suitable for deployment in predicting shipment delays based on various features.

The following steps were taken to ensure a clean and effective modeling process:

1. Data preprocessing and cleaning, including handling missing values and duplicates.
2. Exploratory data analysis to identify key features influencing shipment delays.
3. Model training and evaluation using three different algorithms.
4. Selection of the best model (Logistic Regression) based on performance metrics.


# API Documentation for Shipment Delay Prediction

## Overview

This API predicts shipment delays based on various factors such as weather conditions, traffic conditions, vehicle type, distance, and shipment details. The model uses a trained logistic regression model to predict whether a shipment will be delayed.

The API is built using Flask, and it expects a POST request with the shipment details. The response provides a prediction indicating whether the shipment will be delayed.

## Base URL

1. **Base URL:**  
   `http://localhost:5000`

## API Endpoints

### 1. `/predict` (POST)

This endpoint is used to predict if a shipment will be delayed based on the provided details. The request body should contain information about the shipment, including origin, destination, vehicle type, weather conditions, traffic conditions, shipment date, planned delivery date, and distance.

#### 1.1 Request Format

The request should be sent as a JSON object with the following fields:

```json
{
  "origin": "string",
  "destination": "string",
  "shipment_date": "string",
  "planned_delivery_date": "string",
  "vehicle_type": "string",
  "weather_conditions": "string",
  "traffic_conditions": "string",
  "distance": "float"
}
```

#### 1.2 Fields

1. **origin (string)**  
   The origin city of the shipment.  
   Valid values: `ahmedabad`, `bangalore`, `chennai`, `delhi`, `hyderabad`, `jaipur`, `kolkata`, `lucknow`, `mumbai`, `pune`  
   Example: `"ahmedabad"`

2. **destination (string)**  
   The destination city of the shipment.  
   Valid values: `ahmedabad`, `bangalore`, `chennai`, `delhi`, `hyderabad`, `jaipur`, `kolkata`, `lucknow`, `mumbai`, `pune`  
   Example: `"delhi"`

3. **shipment_date (string)**  
   The date when the shipment is created.  
   Format: `YYYY-MM-DD HH:MM:SS`  
   Example: `"2024-12-20 12:00:00"`

4. **planned_delivery_date (string)**  
   The planned date for the delivery of the shipment.  
   Format: `YYYY-MM-DD HH:MM:SS`  
   Example: `"2024-12-22 12:00:00"`

5. **vehicle_type (string)**  
   The type of vehicle used for the shipment.  
   Valid values: `trailer`, `truck`, `container`, `lorry`, `unknown`  
   Example: `"truck"`

6. **weather_conditions (string)**  
   The weather conditions at the time of shipment.  
   Valid values: `rain`, `storm`, `clear`, `fog`  
   Example: `"rain"`

7. **traffic_conditions (string)**  
   The traffic conditions affecting the shipment.  
   Valid values: `light`, `moderate`, `heavy`  
   Example: `"heavy"`

8. **distance (float)**  
   The distance in kilometers between the origin and destination.  
   Example: `250.5`

#### 1.3 Response Format

The response will be a JSON object with the following structure:

```json
{
  "delayed": "Yes"  // or "No"
}
```

1. **delayed (string)**  
   The prediction result indicating whether the shipment will be delayed.  
   Possible values: `"Yes"`, `"No"`

#### 1.4 Example Request

To send a request to the `/predict` endpoint, use the following details:

```bash
POST /predict
Content-Type: application/json

{
  "origin": "ahmedabad",
  "destination": "delhi",
  "shipment_date": "2024-12-20 12:00:00",
  "planned_delivery_date": "2024-12-22 12:00:00",
  "vehicle_type": "truck",
  "weather_conditions": "rain",
  "traffic_conditions": "heavy",
  "distance": 250.5
}
```

#### 1.5 Example Response

The API will return the following response if the shipment is predicted to be delayed:

```json
{
  "delayed": "Yes"
}
```

### 2. Error Handling

The API will return error messages with an appropriate HTTP status code if the request contains invalid data or if there is an internal error.

1. **400 Bad Request**: If any of the fields are invalid or missing.  
   Example error response:
   ```json
   {
     "error": "Invalid origin. Valid values: ['ahmedabad', 'bangalore', 'chennai', 'delhi', 'hyderabad', 'jaipur', 'kolkata', 'lucknow', 'mumbai', 'pune']"
   }
   ```

2. **500 Internal Server Error**: If an unexpected error occurs while processing the request.  
   Example error response:
   ```json
   {
     "error": "Internal server error message"
   }
   ```

## How to Use the API

1. **Start the Flask Application:**  
   Ensure you have Flask and the necessary dependencies installed. You can install them using pip:

   ```bash
   pip install flask joblib pandas scikit-learn numpy
   ```

2. **Run the Application:**  
   Run the following command in the terminal to start the Flask server:

   ```bash
   python app.py
   ```

   The Flask application will run at `http://localhost:5000`.

3. **Send a POST Request:**  
   Use any HTTP client like `Postman` or `cURL` to send a POST request to `http://localhost:5000/predict` with the required JSON data as described in the request format section.

   Example using `cURL`:

   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
     "origin": "ahmedabad",
     "destination": "delhi",
     "shipment_date": "2024-12-20 12:00:00",
     "planned_delivery_date": "2024-12-22 12:00:00",
     "vehicle_type": "truck",
     "weather_conditions": "rain",
     "traffic_conditions": "heavy",
     "distance": 250.5
   }'
   ```

4. **Interpret the Response:**  
   The API will return a response with either `"Yes"` or `"No"` based on whether the shipment is predicted to be delayed.

## Conclusion

This API allows you to easily predict shipment delays based on various factors. By sending a POST request with the appropriate shipment details, the model will provide a prediction of whether the shipment will be delayed, which can be used to optimize logistics and shipment tracking systems.
