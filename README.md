# EDA of Bankruptcy Data and Building Random Forest, Gradient Boosted Trees, and Extra Trees

The research question of interest is the predict if a firm in a given year has entered bankruptcy proceedings (1) or has not (0). 

The code can be run interactively through a [Google Colab Notebook](https://colab.research.google.com/github/jhancuch/bankruptcy-trees-bagging-boosting/blob/main/script.ipynb).

## Research design and modeling methods
The research design uses three different classifiers, Random Forest, Gradient Boosted Trees and Extra Trees. Examining the data through EDA, I discover I have an unbalanced dataset with only ~3% of observations being bankrupt firms. Thus, when I conduct hyperparameter tuning for Extra Trees through a grid search with a cross validation of k=5, I include potential parameter options of None and balanced to help determine the optimal model parameters.

I additionally utilize both a dataset containing all the variables and a second dataset containing only a subset of variables that have at least a correlation coefficient of |.1| or greater. For each of the three models, I conduct hyperparameter tuning on both the full dataset and subsetted dataset.

## Results and evaluation
| Metric | Random Forest (full) | Random Forest (subset) | Gradient Boosted Trees (full) | Gradient Boosted Trees (subset) | Extra Trees (full) | Extra Trees (subset) |
|---     | ---                  | ---                    | ---                           | ---                             | ---                | ---                  |
| Validation Accuracy | 0.97 | 0.97 | 0.97 | 1.00 | 0.97 | 0.97 |
| Validation TPR | 0.05 | 0.03 | 0.16 | 0.05 | 0.05 | 0.03 |
| Validation TNR | 1.00 | 1.00 | 0.99 | 0.97 | 1.00 | 1.00 |
| Validation (1) Precision | 0.29 | 0.17 | 0.43 | 0.25 | 1.00 | 0.20 |
| Validation (1) Recall | 0.05 | 0.03 | 0.16 | 0.05 | 0.05 | 0.03 |
| Validation (1) F1-Score | 0.09 | 0.05 | 0.24 | 0.09 | 0.10 | 0.05 |
| Validation (0) Precision | 0.97 | 0.97 | 0.98 | 0.97 | 0.97 | 0.97 |
| Validation (0) Recall | 1.00 | 1.00 | 0.99 | | 1.00 | 1.00 |
| Validation (0) F1-Score | 0.99 | 0.98 | 0.99 | 0.98 | 0.99 | 0.99 |

Overall, the run-away winner was the Gradient Boosted Trees tuned model fitted to the subset of the data. Its precision was 54%, recall 19% and had an f1-score of .28. Of all the classification models over I've examined over the course of two weeks, this one performed the best.

## Discusion
The Extra Trees model fitted on the full dataset could have uses. With its precision of 100% and recall of 5%, while it has lots of false negatives, it has no false positives meaning it is conservative with predicting a 1 value. This could be used by individuals trading to avoid any companies that the model predict as going bankrupt due to the near certainty the model provides. It can also provide more assurance with bond discount trading.
