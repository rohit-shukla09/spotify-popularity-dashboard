🎵 Spotify Popularity Analytics & Predictive Modeling Dashboard
An end-to-end data science application engineered to analyze global streaming trends and forecast track popularity using advanced machine learning techniques. This project integrates a memory-optimized preprocessing pipeline, a trained regression model, and an interactive frontend to deliver real-time acoustic insights and geospatial chart analytics.

📌 Executive Summary
This repository contains a full-stack data science solution that bridges raw audio feature datasets with an accessible analytical dashboard. By applying target encoding to high-cardinality categorical variables and utilizing an XGBoost regression model, the system predicts the commercial viability of music tracks based on intrinsic acoustic properties. The interactive UI allows users to manipulate audio parameters, benchmark against genre baselines, and explore artist dominance on a global scale.

🚀 Core Features & Methodology
Predictive Machine Learning Engine: Utilizes an integrated Scikit-Learn Pipeline and ColumnTransformer featuring an XGBRegressor algorithm. The model evaluates user-inputted acoustic features (e.g., danceability, energy, valence, tempo) alongside categorical data (artist, genre) to generate a standardized popularity score (0-100).

Robust Data Engineering: The training script is designed for resource efficiency, processing a massive 1.18 million-row dataset using memory-lean chunking techniques, optimal data type downcasting, and automated data leakage prevention.

Comparative Acoustic Radar: Employs interactive Plotly radar charts to map an individual track’s sonic profile (danceability, energy, acousticness, etc.) directly against aggregated historical baselines for major genres.

Geospatial Trend Analysis: Aggregates and cleans raw global chart data to visualize artist chart appearances worldwide, projecting the output onto a choropleth heatmap.

🛠️ Technical Architecture
Data Engineering & Modeling
Data Manipulation: pandas, numpy (optimized with memory-safe dtypes)

Machine Learning: scikit-learn (Pipelines, TargetEncoder, evaluation metrics), xgboost (Hist-based tree regression)

Serialization: joblib (Model & pipeline artifact export)

Frontend & Visualization
Framework: streamlit (Reactive web application)

Visualization: plotly.express, plotly.graph_objects (Interactive charting and mapping)

📁 Repository Structure
Plaintext
├── DATA ANALYTICS WITH AI/
│   ├── app.py                             # Streamlit dashboard application
│   ├── preprocessing.py                   # Model training & data chunking script
│   ├── requirements.txt                   # Project dependencies
│   ├── .gitignore                         # Excluded large datasets & caches
│   ├── README.md                          # Project documentation
│   ├── spotify_popularity_pipeline.pkl    # Serialized ML pipeline & preprocessor
│   └── cleaned_universal_top_songs.csv    # Processed geospatial chart data
(Note: Raw datasets spotify_full.csv and universal_top_spotify_songs.csv are omitted from the repository due to file size constraints.)

📈 Model Performance
The predictive model was evaluated on a 20% holdout test set containing previously unseen tracks.

Baseline RMSE (Predicting Mean): 16.23

XGBoost Model RMSE: 8.31

Mean Absolute Error (MAE): 6.02

R² Score: 0.738