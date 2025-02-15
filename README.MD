# House Price Prediction

## Overview
This project aims to predict house prices based on various features such as area, number of bedrooms, bathrooms, stories, availability of amenities, and more. The model is built using a **Random Forest Regressor**, which achieved an R² score of **0.7776**. 

## Dataset
The dataset consists of **546 rows** and the following columns:

- `price` (Target Variable)
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`

### Dataset Summary:
- **Maximum House Price:** ₹13,300,000
- **Minimum House Price:** ₹1,750,000

## Model Performance
- **Mean Squared Error (MSE):** 859,345,069,686.5701
- **Mean Absolute Error (MAE):** 671,879.59
- **Percentage Error:** ~14.10%
- **Best Model:** Random Forest Regressor (R² = 0.7776)

## Data Preprocessing
- Handled categorical variables using encoding.
- Scaled numerical features.
- Removed missing or inconsistent values.
- Used a train-test split strategy for model evaluation.

## Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn** (for visualization)
- **Scikit-learn** (for model training)
- **Random Forest Regressor**

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required libraries. You can install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/house-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Run the script:
   ```bash
   python house_price_prediction.py
   ```

## Results & Insights
- The model successfully predicts house prices with a reasonable accuracy.
- Feature importance analysis showed that **area, bedrooms, and parking** have a significant impact on price.
- Further improvements can be made by tuning hyperparameters and experimenting with advanced models.

## Future Improvements
- Implement feature engineering to extract new insights.
- Experiment with boosting models like **XGBoost** or **LightGBM**.
- Deploy the model as a web application for real-world usage.

## Author
- **Bhavesh Nikam**

## License
This project is licensed under the MIT License.

