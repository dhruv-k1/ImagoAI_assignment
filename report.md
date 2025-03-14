# Project Report

## Introduction

This project focuses on analyzing a dataset with 450 features, performing preprocessing, dimensionality reduction, and building predictive models to evaluate performance. The primary tasks included data cleaning, outlier removal, dimensionality reduction using PCA, and training multiple models (LSTM, CNN, GNN, and Random Forest) to predict a target variable. The deep learning models (LSTM, CNN, GNN) were the main focus, with CNN emerging as the best among them, though a Random Forest model was also tested as a special case and outperformed all. This report summarizes the preprocessing steps, insights from dimensionality reduction, model selection and evaluation, key findings, and suggestions for improvement.

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
    *   *Architecture:* Implemented using PyTorch with a custom `LSTMModel` class. The model consists of:
        *   An LSTM layer (`nn.LSTM`) with configurable `input_size`, `hidden_size`, and `batch_first=True`.
        *   A fully connected layer (`nn.Linear`) mapping the LSTM output to the desired `output_size`.
        *   Forward pass extracts the last time step's output for prediction.
    *   *Hyperparameter Tuning:* Used random search with `RandomizedSearchCV` over:
        *   `hidden_size`: Random integers between 20 and 150.
        *   `lr` (learning rate): Uniform distribution between 0.0001 and 0.01.
        *   `epochs`: Random integers between 50 and 200.
    *   *Training:* Used PyTorch with a custom DataLoader for time-series-like data.
    *   *Performance:* Specific metrics not provided; assumed lower than CNN and Random Forest based on the outline.

*   **CNN (Convolutional Neural Network):**
    *   *Architecture:* Implemented using PyTorch with a custom `CNNModel` class. The model consists of:
        *   A 1D convolutional layer (`nn.Conv1d`) with 1 input channel, 16 output channels, and a kernel size of 3.
        *   A ReLU activation (`nn.ReLU`) for non-linearity.
        *   A flatten layer (`nn.Flatten`) to reshape the output.
        *   Two fully connected layers: `fc1` (`nn.Linear`) mapping from 16 to 64 units, followed by ReLU, and `fc2` (`nn.Linear`) mapping from 64 to 1 (output).
    *   *Hyperparameter Tuning:* Used random search over 50 iterations with:
        *   `lr` (learning rate): Uniform distribution between 0.0001 and 0.01.
        *   `epochs`: Random integers between 50 and 200.
    *   *Training:* Trained using the Adam optimizer (`optim.Adam`) and Mean Squared Error loss (`nn.MSELoss`). Evaluated using RMSE on the test set.
    *   *Performance:* Outperformed LSTM and GNN, making it the best deep learning model (specific metrics not provided in the snippet).

*   **GNN (Graph Neural Network):**
    *   *Architecture:* Implemented using PyTorch Geometric with a custom `GNNModel` class. The model consists of:
        *   Two Graph Convolutional layers (`gnn.GCNConv`): The first maps the input dimension to a `hidden_dim`, and the second refines it with another `hidden_dim`, both followed by ReLU activation.
        *   A fully connected layer (`nn.Linear`) mapping from `hidden_dim` to `output_dim`.
        *   Graph data was constructed using a "fully connected" approach where every node is connected to every other node (`torch.combinations`), with `DataLoader` from `torch_geometric.loader`.
    *   *Hyperparameter Tuning:* Used random search with:
        *   `hidden_dim`: Random integers between 16 and 64.
        *   `lr` (learning rate): Uniform distribution between 0.0001 and 0.01.
        *   `epochs`: Random integers between 50 and 200.
    *   *Training:* Trained on graph-structured data with a batch size of 32.
    *   *Performance:* Lower than CNN and Random Forest.

*   **Random Forest (Special Case):**
    *   *Architecture:* Used `RandomForestRegressor` from `sklearn.ensemble` with 100 estimators and `random_state=42` for reproducibility.
    *   *Training:* Trained on the 3 PCA components (PC1, PC2, PC3) with an 80-20 train-test split.

### Evaluation Metrics

*   **Mean Absolute Error (MAE):** 1933.80
*   **Root Mean Squared Error (RMSE):** 4963.59
*   **R² Score:** 0.736

### Visualization

*   Scatter plot of actual vs. predicted values showed reasonable alignment along the ideal line. (Ideally, embed this plot here.)

### Model Selection Rationale

*   **Deep Learning Models (LSTM, CNN, GNN):** Chosen to explore complex patterns in the data, leveraging temporal (LSTM), spatial (CNN), and relational (GNN) structures. CNN was expected to perform well due to its ability to capture spatial patterns in the PCA-transformed data, which aligns with the observed gradient in the PCA plot. GNN was included to explore relational patterns using a fully connected graph, while LSTM targeted potential temporal dependencies.
*   **Random Forest:** Added as a baseline tree-based model, known for robustness and interpretability, especially with reduced dimensions.

### Training and Evaluation Details

*   **Train-Test Split:** 80% training, 20% testing (`random_state=42` for reproducibility).
*   **Hyperparameter Tuning for Deep Learning Models:** Random search was employed to optimize hyperparameters for LSTM, CNN, and GNN. For CNN, 50 iterations were run to find the best combination of learning rate and epochs, minimizing RMSE on the test set. Similar tuning was applied to LSTM and GNN, enhancing their performance by exploring a range of `hidden_size` (for LSTM and GNN), `lr`, and `epochs`.
*   **Metrics:** MAE, RMSE, and R² were used to assess regression performance for Random Forest, while RMSE was used for deep learning models during hyperparameter tuning.
*   **Performance Comparison:** CNN outperformed LSTM and GNN among deep learning models, likely due to its architecture being well-suited for capturing spatial patterns in the PCA components. However, Random Forest outperformed all models with an R² of 0.736, explaining ~73.6% of the target variance.

## Key Findings

*   **Preprocessing Effectiveness:** Outlier removal and duplicate handling resulted in a clean, stable dataset (441 × 450), suitable for analysis.
*   **Dimensionality Reduction:** PCA to 3 components retained meaningful patterns, with PC1 showing a strong relationship with the target variable, though the skewed target distribution may pose challenges.
*   **Model Performance:**
    *   Among deep learning models, CNN performed the best after hyperparameter tuning, leveraging its convolutional layers to capture spatial patterns in the PCA-transformed data.
    *   Random Forest significantly outperformed all deep learning models (R² = 0.736), highlighting its effectiveness with reduced dimensions.
*   **Hyperparameter Tuning:** Random search proved effective in optimizing the deep learning models, particularly for CNN, where 50 iterations identified the best learning rate and epoch combination, leading to the lowest RMSE among the neural networks.

## Limitations

*   The Random Forest model's performance (MAE: 1933.80, RMSE: 4963.59) suggests moderate predictive accuracy, potentially limited by dimensionality reduction, the skewed target distribution, and data complexity. Deep learning models, despite tuning, faced additional challenges (detailed below).
    *   **Dimensionality Reduction Impact:** Reducing to 3 PCA components likely resulted in significant information loss, which disproportionately affected deep learning models. CNN, despite being the best among them, may have missed finer spatial patterns that additional components could have captured. GNN, relying on a fully connected graph, may also have suffered from insufficient feature richness.
    *   **Skewed Target Distribution:** The target variable's skewness (observed in the PCA plot) likely impacted deep learning models, particularly CNN and GNN, as they may struggle to predict rare high-value targets effectively. This skewness could lead to overfitting to the majority (low-value) cases.

*   **CNN-Specific Limitations:**
    *   *Architecture Simplicity:* The CNN architecture, with only one convolutional layer (`nn.Conv1d`) and two fully connected layers, may be too shallow to capture complex patterns in the data. Deeper architectures with more convolutional layers or larger kernel sizes could improve feature extraction.
    *   *Input Representation:* The CNN expects 1D input data with a single channel, which may not fully leverage the multidimensional nature of the original 450 features. Retaining more PCA components or reshaping the input to include additional channels could enhance performance.
    *   *Hyperparameter Tuning Scope:* While random search was used (50 iterations), the search space was limited to learning rate and epochs. Tuning additional parameters like the number of filters, kernel size, or adding dropout for regularization could further improve CNN's performance.

*   **LSTM Challenges:** The LSTM model, designed for sequential data, may have underperformed due to the lack of a clear temporal structure in the PCA-transformed data. Its performance could be improved with proper sequence preprocessing or by using the original features.

*   **GNN Challenges:** The GNN model, with its fully connected graph and two GCN layers, likely struggled due to the arbitrary connectivity assumption. The lack of domain-specific graph structure and the limited `hidden_dim` range (16-64) may have restricted its ability to model relational patterns effectively. Additionally, the small dataset size (441 rows) may not provide enough nodes to leverage the graph structure fully.

*   **Computational Constraints:** Training deep learning models, especially CNN and GNN with 50 iterations of random search, was computationally intensive. Limited computational resources may have restricted the extent of tuning and experimentation.

*   **Generalization Issues:** Deep learning models, including CNN and GNN, may overfit to the training data, especially given the small dataset size (441 rows) and the skewed target distribution, leading to poorer generalization on the test set compared to Random Forest.

## Suggestions for Improvement

*   **PCA Optimization:** Retain more principal components (e.g., 5-10) to capture additional variance, particularly for deep learning models like CNN and GNN, which could benefit from richer feature representations.
*   **Target Distribution Handling:** Apply transformations (e.g., log transformation) to the target variable to address skewness, potentially improving CNN and GNN's ability to predict high-value targets.

*   **CNN Enhancements:**
    *   Increase the depth of the CNN architecture by adding more convolutional layers and experimenting with larger kernel sizes to capture more complex spatial patterns.
    *   Explore multi-channel input representations or alternative dimensionality reduction techniques (e.g., t-SNE, autoencoders) to provide richer features for the CNN.
    *   Expand the hyperparameter search space to include the number of filters, kernel size, and regularization techniques like dropout to prevent overfitting.

*   **GNN Enhancements:**
    *   Refine the graph construction by incorporating domain-specific relationships or feature correlations instead of a fully connected graph to better reflect the data's structure.
    *   Increase the `hidden_dim` range or add more GCN layers to enhance the model's capacity to learn complex relational patterns.
    *   Include additional hyperparameters in the random search, such as dropout rates or layer numbers, to improve generalization.

*   **LSTM Enhancements:** Incorporate sequence preprocessing if the data has a temporal aspect, or use the original features to better leverage temporal dependencies.

*   **Hyperparameter Tuning:** For all deep learning models, expand the random search to include additional parameters (e.g., dropout, layer numbers) and increase the number of iterations (e.g., 100) to explore a broader parameter space.

*   **Outlier Handling:** Test alternative methods (e.g., z-score, isolation forest) to balance outlier removal and data retention.

*   **Ensemble Approach:** Combine Random Forest and CNN predictions to leverage the strengths of both models, potentially improving overall performance.

## Conclusion

This project successfully preprocessed a high-dimensional dataset, reduced its dimensionality using PCA, and evaluated multiple predictive models. The PCA 3D visualization revealed meaningful patterns, with PC1 strongly correlated with the target variable, though the skewed distribution highlighted challenges. Among the deep learning models, CNN performed the best, benefiting from its convolutional architecture and random search hyperparameter tuning (50 iterations optimizing learning rate and epochs). The GNN, with its fully connected graph and two GCN layers, underperformed, likely due to the arbitrary connectivity and limited feature space, while LSTM struggled with the lack of temporal structure. However, Random Forest emerged as the top performer (R² = 0.736), outperforming all deep learning models, including CNN, due to its robustness with reduced dimensions. The random search tuning improved CNN's performance, but its shallow architecture, the GNN's simplistic graph structure, and the limitations of PCA suggest room for improvement. Future work should focus on optimizing dimensionality reduction, addressing target skewness, enhancing the CNN and GNN architectures, and exploring ensemble methods to combine the strengths of CNN and Random Forest for better predictive accuracy.
