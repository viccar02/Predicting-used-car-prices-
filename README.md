# Used Car Price Prediction

This project predicts used car prices using a machine learning pipeline built on structured vehicle listing data. The notebook covers data cleaning, feature engineering, exploratory analysis, model comparison, hyperparameter tuning, and model interpretation with SHAP values. [file:5]

## Project Overview

The notebook uses a used-car dataset loaded directly from a public GitHub CSV and builds regression models to estimate listing prices. The raw dataset contains 4,009 rows and 12 columns, including brand, model, model year, mileage, fuel type, engine, transmission, exterior color, interior color, accident history, title status, and price. [file:5]

The workflow then cleans and transforms these fields into a modeling-ready dataset with engineered features such as horsepower, turbo, car age, normalized mileage, cleaned color categories, and numeric target variables. A final cleaned modeling table shown in the notebook contains 3,764 rows. [file:5]

## Data

The original dataset includes the following core features:
- `brand`
- `model`
- `model_year`
- `milage`
- `fuel_type`
- `engine`
- `transmission`
- `ext_col`
- `int_col`
- `accident`
- `clean_title`
- `price` [file:5]

The notebook engineers additional variables such as:
- `turbo`
- `hp`
- `car_age`
- `price2`
- `ext_col_cleaned`
- `int_col_cleaned` [file:5]

It also standardizes raw text-based values, for example converting mileage and price into numeric form, reducing transmission labels to broader categories, and grouping color values into cleaner classes. [file:5]

## Feature Engineering

Several preprocessing steps are visible in the notebook:

- Mileage strings such as `"51,000 mi."` are converted into numeric mileage values. [file:5]
- Price strings such as `"$10,300"` are converted into numeric targets. [file:5]
- Horsepower is extracted from the engine text field, and a `turbo` indicator is created from engine descriptions. [file:5]
- `car_age` is derived from `model_year`. [file:5]
- Exterior and interior colors are grouped into cleaned categories like Black, White, Gray, Blue, Red, and Other. [file:5]
- Transmission values are simplified into categories such as Automatic, Manual, and Other. [file:5]

For example, the cleaned exterior color counts include Black (861), White (779), Gray (462), Silver (364), Blue (337), and Green (67), while cleaned interior colors are dominated by Black (1,908), Beige (525), and Gray (461). Transmission is mostly Automatic at 3,397 listings, followed by Manual at 357 and Other at 11. [file:5]

## Modeling

The notebook compares several regression models:
- Dummy Regressor
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor [file:5]

The reported test results are:

| Model | Performance |
|---|---|
| XGBoost | RMSE 8,580.50, MAPE 23.39%, R² 0.840 [file:5] |
| Ridge | RMSE 8,673.01, MAPE 32.86%, R² 0.837 [file:5] |
| Random Forest | RMSE 8,845.40, MAPE 23.46%, R² 0.830 [file:5] |
| Dummy | RMSE 21,540.29, MAPE 107.69%, R² -0.006 [file:5] |

These results show that XGBoost performs best in the notebook, with the lowest RMSE and highest \(R^2\) among the tested models. The dummy baseline performs very poorly, which confirms that the engineered features and trained models capture substantial price variation. [file:5]

## Explainability

The notebook uses SHAP to interpret the XGBoost model. The most important individual features by mean absolute SHAP value are horsepower, mileage, and car age, followed by brand and fuel-related variables. [file:5]

Some of the top SHAP-ranked features shown are:
- `num__hp`: 7644.97
- `num__milage`: 7225.38
- `num__car_age`: 6265.25
- `cat__brand_Porsche`: 1182.94
- `cat__fuel_type_Diesel`: 768.37
- `bool__accident`: 366.16 [file:5]

When grouped into broader parent categories, the most influential groups are horsepower, mileage, car age, brand, model, and fuel. This makes the model easier to explain in business terms, not just one-hot encoded feature names. [file:5]

## Requirements

The notebook imports the following main libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `plotly`
- `optuna`
- `scikit-learn`
- `xgboost`
- `shap`
- `folium`
- `joblib` [file:5]

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy plotly optuna scikit-learn xgboost shap folium joblib
```

## How to Run

1. Open `M1_assignment_EUmix.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. Install the required Python packages.
3. Run the notebook from top to bottom.
4. The dataset is loaded automatically from the public GitHub CSV link in the notebook, so no manual file download is required. [file:5]

The notebook notes that:
- seeds are set to 42 for reproducibility
- the trained XGBoost model is saved for later use in an application pipeline
- Plotly graphs render in Colab by default, and VS Code users may need to switch the renderer from `colab` to `browser` [file:5]

## Outputs

The notebook produces:
- a cleaned regression dataset
- exploratory tables and visualizations
- trained benchmark and advanced models
- an exported XGBoost model artifact
- SHAP-based feature importance outputs [file:5]

## Key Takeaways

This project shows that a relatively compact structured dataset can support strong price prediction when combined with thoughtful feature engineering. The strongest drivers of price in this notebook are mechanical performance, vehicle age, mileage, and brand effects, with XGBoost delivering the best overall predictive accuracy. [file:5]
