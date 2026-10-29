Toyota Camry Auction Price Prediction

Predicting used Toyota Camry auction sale prices in USD using vehicle specifications, mileage, seller information, and damage data.

## Data and Workflow

- Audited dataset: 2,648 records covering model years 2020–2026.
- Random split: 70% training, 15% validation, 15% testing.
- Preprocessing: median imputation, missing-value indicators, numeric scaling, and one-hot encoding.
- Models: Linear Regression, Ridge, and Lasso, with five-fold cross-validation for regularization tuning.
- Preprocessing stays inside the pipeline to prevent leakage during fitting and cross-validation.

## Results

Linear Regression was selected using validation results, then refitted on training and validation data.

MAE: $1,961.09
RMSE: $2,567.77
R²: 0.585

Follow-up experiments showed no meaningful improvement, so the original baseline was retained.

## Limitations

Results reflect a historical random split, not current-market forecasting. Mileage placeholders, inconsistent source labels, and potential repeated vehicles require verification. MAE is an average error, not a prediction interval. Further model development should use fresh held-out data for independent evaluation.
