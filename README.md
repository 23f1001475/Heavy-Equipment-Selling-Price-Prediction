# Heavy Equipment Selling Price Prediction

Predicting the selling price of used heavy machinery based on its specifications, usage, and sale details.

Built for the Kaggle competition **[Heavy Equipment Selling Price Prediction Challenge](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge)**.

|                      |                                                    |
| -------------------- | -------------------------------------------------- |
| **Task**             | Regression — predict `TargetValue`, the sale price |
| **Metric**           | RMSLE (RMSE on `log1p(price)`)                     |
| **Validation RMSLE** | `0.2034` — best single model (LightGBM)            |
| **Kaggle Score**     | `0.19862`                                          |
| **Kaggle Notebook**  | [View Notebook](https://www.kaggle.com/code/vivek23f1001475/heavy-equipment-price-prediction/edit)    |

## Approach

### 1. Data Audit

* Inspected column types, date ranges, and duplicates.
* Checked missing values, skewness, and cardinality.
* Ignored ID columns when checking for duplicates.

### 2. Exploratory Data Analysis

* Analyzed the target distribution and applied a log transformation consistent with the RMSLE metric.
* Performed univariate and bivariate analysis.
* Compared Pearson and Spearman correlations.

### 3. Data Cleaning

* Detected Excel errors such as `#NAME?` and converted them to missing values.
* Identified placeholder years such as `1001`.
* Fixed case-duplicate categorical values.
* Clipped outliers using IQR on measurement columns only.

### 4. Feature Engineering

Created features including:

* Sale date components
* Equipment age
* Log-transformed hours
* Hours per year
* Age × hours interaction
* Number of filled specifications
* Missing-value indicators

Anonymous `colN` columns were decoded using `metadata.csv`.

### 5. Preprocessing

* Merged rare categories.
* Applied target encoding to high-cardinality categorical features.
* Used ordinal and one-hot encoding for low-cardinality features.
* Used native missing-value handling where supported by tree-based models.

### 6. Modelling

Five baseline models were evaluated:

* Ridge Regression
* Gradient Boosting
* HistGradientBoosting
* LightGBM
* XGBoost

The three strongest boosted-tree models were then tuned using `RandomizedSearchCV`.

### 7. Model Interpretation

* Analyzed the best hyperparameters.
* Examined parameter sensitivity.
* Used LightGBM gain-based feature importance to understand the most influential features.

### 8. Ensemble

Created an inverse-RMSLE weighted ensemble of the three tuned models in log space and re-fitted the models on the combined training and validation data.

## Results

| Model                | Baseline RMSLE | Tuned RMSLE |
| -------------------- | -------------: | ----------: |
| Ridge                |         0.2865 |           — |
| Gradient Boosting    |         0.2338 |           — |
| HistGradientBoosting |         0.2157 |      0.2076 |
| XGBoost              |         0.2160 |      0.2062 |
| **LightGBM**         |         0.2285 |  **0.2034** |
| Weighted Ensemble    |              — |      0.2037 |

**Validation RMSLE:** 20% hold-out validation set. Lower is better.

Hyperparameter tuning used 3-fold `RandomizedSearchCV`:

* 100 candidates for HistGradientBoosting
* 100 candidates for LightGBM
* 20 candidates for XGBoost

## Key Findings

### LightGBM benefited most from tuning

LightGBM improved from **0.2285 → 0.2034**, a reduction of **0.0251 RMSLE**, moving from fourth among the baseline models to the best-performing model.

Other improvements:

* HistGradientBoosting: `0.2157 → 0.2076`
* XGBoost: `0.2160 → 0.2062`

### Ensemble vs. best single model

The weighted ensemble achieved `0.2037`, slightly worse than the best single LightGBM model at `0.2034`.

This suggests that the tuned models make highly similar predictions and errors, so blending provides very little additional benefit.

The final submission nevertheless uses the ensemble to reduce dependence on a single model.

### Most important features

According to LightGBM gain-based feature importance:

| Feature                    | Gain Share |
| -------------------------- | ---------: |
| `ProductConfigID`          |        46% |
| `Spec_FullDescriptor`      |        32% |
| `EquipmentAge`             |       5.3% |
| `SaleYear`                 |       3.9% |
| `FunctionalClassification` |       2.4% |

`ProductConfigID` and `Spec_FullDescriptor` account for a large proportion of the model's gain, suggesting that equipment configuration and product identity are strong indicators of selling price.

> **Note:** Gain-based importance can favor high-cardinality features, so these results should be treated as directional rather than definitive.

### Engineered features

`EquipmentAge` ranked **3rd out of 74 features**, ahead of the raw `ManufactureYear`.

Other useful engineered features included:

* `HoursPerYear` — rank 16
* `AgeXHours` — rank 17

Interestingly, **50 of the 74 features contributed less than 0.1% of total gain**, indicating that much of the predictive power is concentrated in a relatively small set of features.

### Data quality issues

The analysis identified several data-quality problems:

* **14,719 rows (10.6%)** contained the placeholder year `1001` in `ManufactureYear`.
* Excel errors such as `#NAME?` were detected programmatically and converted to missing values.
* Duplicate categorical values caused by inconsistent capitalization were standardized.

### Overfitting

The difference between training and validation RMSLE remained relatively small across the baseline models, approximately **0.01–0.017**, suggesting limited overfitting under the chosen validation setup.

## Repository Structure

```text
heavy-equipment-price-prediction/
│
├── notebooks/
│   └── heavy-equipment-price-prediction.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook contains the complete analysis and modelling pipeline, including outputs.

## How to Run

### 1. Download the competition data

Download the data from the [Kaggle competition page](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge/data).

The competition data is **not included in this repository**.

### 2. Create the data directory

Place the following files inside a `data/` folder:

```text
data/
├── train.csv
├── test.csv
├── metadata.csv
└── sample_submission.csv
```

### 3. Configure the data path

In the first notebook cell, set:

```python
DATA_DIR = "../data"
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the notebook

```bash
jupyter notebook notebooks/heavy-equipment-price-prediction.ipynb
```

> **Note:** Hyperparameter tuning can be computationally expensive. For a quick test run, reduce `N_ITER_HGB`, `N_ITER_LGBM`, and `N_ITER_XGB` in the first notebook cell.

## Limitations & Future Improvements

### Time-based validation

The current approach uses a random train/validation split. A time-based split could provide a more realistic estimate if the competition's test data represents future sales.

### Sale history features

Historical sales information could potentially improve predictions.

* 23.3% of test `AssetID`s also appear in the training data.
* 12,857 assets were sold more than once in the training data.

This creates an opportunity to engineer historical asset-level features.

### Text features

`Spec_FullDescriptor` could be explored using dedicated text-processing techniques such as TF-IDF or embeddings.

### Native categorical handling

Future experiments could compare the current encoding approach against models with native categorical handling, such as LightGBM and CatBoost.

## Author

**Vivek Mittal**

* [LinkedIn](https://www.linkedin.com/in/vivek-mittal-574a31250/)
* [GitHub](https://github.com/23f1001475)