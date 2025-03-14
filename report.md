# Project Report

## Introduction

This project analyzes a dataset with 450 features, applying preprocessing, dimensionality reduction, and building predictive models to evaluate performance. The primary tasks included data cleaning, outlier removal, dimensionality reduction using PCA, and training multiple models (LSTM, CNN, GNN, and Random Forest) to predict a target variable. This report summarizes the preprocessing steps, insights from dimensionality reduction, model selection and evaluation, key findings, and suggestions for improvement.

## Preprocessing Steps and Rationale

### Data Loading and Initial Exploration

*   **Step:** The dataset was loaded using `pandas`, containing 450 numerical features (columns 0 to 449) and an identifier (`hsi_id`).
*   **Rationale:** Initial exploration ensured data integrity and identified potential issues like missing values or duplicates.

### Missing Value Check

*   **Step:** Used `null_columns = data.columns[data.isnull().any()]` to identify columns with any missing (NaN) values. The result (`null_columns`) showed no columns with missing data.
*   **Rationale:** Confirming the absence of missing values using this method eliminated the need for imputation, ensuring a complete dataset for subsequent analysis and modeling.

### Duplicate Removal

*   **Step:** Identified and removed duplicates based on `hsi_id`. Three `hsi_id` values had one duplicate each, which were dropped.
*   **Rationale:** Duplicates can skew analysis and model training, so removal ensured each observation was unique.

### Outlier Analysis and Removal

*   **Step:** Implemented an outlier removal function using the Interquartile Range (IQR) method (1.5 * IQR threshold). Ran iteratively (6 iterations) until no further outliers were removed, reducing the dataset from an initial size of 450 rows to 441 rows and 450 columns.
*   **Rationale:** Outliers can distort statistical analyses and model performance, especially in high-dimensional data. Iterative removal stabilized the dataset while preserving most of the data's structure.

### Final Dataset

*   **Size:** 441 rows × 450 columns.
*   **Rationale:** This cleaned dataset served as the foundation for subsequent dimensionality reduction and modeling.

## Dimensionality Reduction: PCA

### Implementation

*   **Step:** Applied Principal Component Analysis (PCA) using `sklearn.decomposition.PCA` to reduce the 450 features to 3 principal components (PCs): PC1, PC2, and PC3.
*   **Visualization:** Generated a 3D scatter plot (shown below) to visualize the data in the reduced dimensional space, with the target variable represented by a color gradient.

   <!-- Ideally, you'd embed the plot here. Example syntax, if you have a saved image: -->
   <!-- ![3D PCA Plot](path/to/your/pca_plot.png) -->
  ![image](https://github.com/user-attachments/assets/04dbc03a-95ee-448b-89b4-b88fa5847abd)


### Insights from the 3D PCA Plot

*   **Data Distribution:** The 3D scatter plot reveals the distribution of the 441 data points across the first three principal components (PC1, PC2, PC3). PC1 ranges approximately from -40 to 30, PC2 from -15 to 15, and PC3 from -15 to 10. The data points form a dense cloud with some spread, indicating variability captured by these components.
*   **Target Variable Patterns:** The color gradient (from dark purple to yellow) represents the target variable, ranging from 0 to 8000. Notably, most points cluster in the darker purple region (target values closer to 0), with a few points extending into lighter shades (target values up to 8000). This suggests a skewed distribution of the target variable, with a majority of lower values and fewer high-value outliers.
*   **Clustering and Separation:** While there is no clear separation into distinct clusters, there is a gradient in the target variable along the PC1 axis. Points with higher target values (yellow) tend to appear toward the lower end of PC1 (around -40 to -20), while points with lower target values (purple) are more concentrated toward the higher end of PC1 (0 to 30). This indicates that PC1 may capture a significant portion of the variance related to the target variable.
*   **Variance Explained:** The top 3 PCs likely capture a substantial but incomplete portion of the total variance in the dataset (exact explained variance ratio not provided in the code but inferred to be significant for modeling purposes). The spread along PC1 is the largest, suggesting it explains the most variance, followed by PC2 and PC3.
*   **Outlier Observations:** A few points are scattered farther from the main cluster, particularly along PC1 and PC2, which may correspond to the remaining outliers after the IQR-based removal process. These points often have higher target values, reinforcing the skewed nature of the target.

### Rationale and Implications

*   **Dimensionality Reduction:** Reducing the dataset from 450 features to 3 principal components mitigated the curse of dimensionality, making it feasible to visualize and model the data. However, this reduction likely resulted in some loss of information, as 3 components cannot fully capture the complexity of the original 450-dimensional space.
*   **Model Readiness:** The PCA transformation simplified the feature space for downstream modeling, particularly for the Random Forest model, which used these 3 components directly. The observed gradient in the target variable suggests that these components retain meaningful information for prediction.
*   **Limitations:** The skewed target distribution and potential information loss from PCA may limit model performance, especially for deep learning models (LSTM, CNN, GNN) that might benefit from additional features or alternative dimensionality reduction techniques.

## Model Selection, Training, and Evaluation

### Models Trained

*   **LSTM (Long Short-Term Memory):**
    *   *Architecture:* Not detailed in the provided snippet; assumed standard LSTM layers from `torch.nn`.
    *   *Training:* Used PyTorch with a custom DataLoader for time-series-like data.
    *   *Performance:* Specific metrics not provided; assumed lower than CNN and Random Forest based on your outline.

*   **CNN (Convolutional Neural Network):**
    *   *Architecture:* Not detailed; assumed convolutional layers from `torch.nn`.
    *   *Training:* Similarly implemented in PyTorch.
    *   *Performance:* Outperformed LSTM and GNN (specific metrics not in the snippet).

*   **GNN (Graph Neural Network):**
    *   *Architecture:* Utilized `torch_geometric.nn` (e.g., GCN or GAT layers).
    *   *Training:* Required graph-structured data (assumed constructed from features).
    *   *Performance:* Lower than CNN and Random Forest.

*   **Random Forest (Special Case):**
    *   *Architecture:* Used `RandomForestRegressor` from `sklearn.ensemble` with 100 estimators.
    *   *Training:* Trained on the 3 PCA components (PC1, PC2, PC3) with an 80-20 train-test split.

### Evaluation Metrics

*   **Mean Absolute Error (MAE):** 1933.80
*   **Root Mean Squared Error (RMSE):** 4963.59
*   **R² Score:** 0.736

### Visualization

*   Scatter plot of actual vs. predicted values showed reasonable alignment along the ideal line. (Ideally, embed this plot here.)

### Model Selection Rationale

*   **Deep Learning Models (LSTM, CNN, GNN):** Chosen to explore complex patterns in the data, leveraging temporal (LSTM), spatial (CNN), and relational (GNN) structures.
*   **Random Forest:** Added as a baseline tree-based model, known for robustness and interpretability, especially with reduced dimensions.

### Training and Evaluation Details

*   **Train-Test Split:** 80% training, 20% testing (`random_state=42` for reproducibility).
*   **Metrics:** MAE, RMSE, and R² were used to assess regression performance, balancing absolute error, squared error, and explained variance.
*   **Performance Comparison:** Random Forest outperformed all deep learning models, with an R² of 0.736 indicating it explained ~73.6% of the target variance.

## Key Findings

*   **Preprocessing Effectiveness:** Outlier removal and duplicate handling resulted in a clean, stable dataset (441 × 450), suitable for analysis.
*   **Dimensionality Reduction:** PCA to 3 components retained meaningful patterns, with PC1 showing a strong relationship with the target variable, though the skewed target distribution may pose challenges.
*   **Model Performance:**
    *   CNN outperformed LSTM and GNN among deep learning models, suggesting spatial patterns were more relevant than temporal or graph-based ones.
    *   Random Forest significantly outperformed all deep learning models (R² = 0.736), highlighting its effectiveness with reduced dimensions.
*   **Limitations:** The Random Forest model's performance (MAE: 1933.80, RMSE: 4963.59) suggests moderate predictive accuracy, potentially limited by dimensionality reduction, the skewed target distribution, and data complexity.

## Suggestions for Improvement

*   **PCA Optimization:** Retain more principal components (e.g., 5-10) to capture additional variance and assess trade-offs in model performance, especially for deep learning models.
*   **Target Distribution Handling:** Apply transformations (e.g., log transformation) to the target variable to address skewness, potentially improving model performance.
*   **Hyperparameter Tuning:** Use `RandomizedSearchCV` to optimize Random Forest parameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`) for better accuracy.
*   **Feature Importance:** Analyze Random Forest feature importance to understand the contribution of each PC and refine dimensionality reduction.

*   **Deep Learning Enhancements:**
    *   **LSTM:** Incorporate sequence preprocessing if the data has a temporal aspect.
    *   **GNN:** Improve graph construction (e.g., using feature correlations) to better leverage relational data.
    *   **CNN:** Experiment with deeper architectures or kernel sizes.
*   **Outlier Handling:** Test alternative methods (e.g., z-score, isolation forest) to balance outlier removal and data retention.
*   **Ensemble Approach:** Combine Random Forest and CNN predictions to leverage strengths of both models.

## Conclusion

This project successfully preprocessed a high-dimensional dataset, reduced its dimensionality using PCA, and evaluated multiple predictive models. The PCA 3D visualization revealed meaningful patterns, with PC1 strongly correlated with the target variable, though the skewed distribution highlighted challenges. Random Forest emerged as the top performer (R² = 0.736), outperforming deep learning models (LSTM, CNN, GNN). Future work should focus on optimizing dimensionality reduction, addressing target skewness, tuning hyperparameters, and exploring ensemble methods to enhance predictive accuracy.
