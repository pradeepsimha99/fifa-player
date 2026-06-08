## Model Comparison and Feature Leakage Analysis

Three machine learning models were evaluated for predicting FIFA player overall ratings.

| Model                             | MAE   | R²    |
| --------------------------------- | ----- | ----- |
| Linear Regression                 | 1.599 | 0.906 |
| Random Forest                     | 0.102 | 0.998 |
| Random Forest (Without Potential) | 0.141 | 0.997 |

The Random Forest model significantly outperformed Linear Regression, achieving an R² score above 0.99.

To investigate potential feature leakage, the `potential` attribute was removed and a second Random Forest model was trained. Although model performance decreased slightly, the R² score remained above 0.997.

This suggests that attributes such as dribbling, defending, shooting, passing, and physical ability provide sufficient information to accurately estimate a player's overall FIFA rating.

The experiment demonstrates the importance of feature selection and model validation when building machine learning systems.