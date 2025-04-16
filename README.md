# 3D-Printing-Prediction-System-with-ML-

## [ 1 ] Main questions

### WHAT
This project aims to use machine learning to predict the roughness, elongation, and tensile strength of 3D printed objects based on the printing parameters. These predictions can help optimize the printing process and improve the final product quality.

### WHY
Traditional testing of 3D printed objects for mechanical properties is time-consuming and costly. By predicting these properties using machine learning, we can automate the process and reduce costs significantly.

### HOW
This system uses a dataset of 3D printing parameters (such as temperature, speed, material type, etc.) to train a linear regression model. This model is then used to predict the roughness, elongation, and tensile strength of new prints in real time through a ROS node.

## [ 2 ]  **Concept of Operations (CoO)**

**Inputs:**
- 3D print parameters (temperature, speed, material type, infill pattern, etc.)

**Outputs:**
- Predicted properties (roughness, elongation, tensile strength)

## [ 3 ]  Where ML Contributes:**
- In the "Prediction" block of the diagram, the machine learning model takes in the print parameters and outputs predictions for the material properties.

## [ 4 ]  **Software Stack & Technologies Used**
- **Python** for scripting
- **scikit-learn** for machine learning algorithms
- **ROS** for real-time predictions and integration with hardware
- **pandas, numpy** for data handling
- **joblib** for saving/loading models
- **ipywidgets** for UI integration
- **matplotlib** for data visualization

## [ 5 ] **ML Expectations**
The ML model is expected to provide accurate predictions for the roughness, elongation, and tensile strength based on input parameters. These predictions should help optimize the printing process and ensure high-quality 3D prints.

## [ 6 ]  **Where Do You Get Data?**
The data is collected from a variety of 3D prints, with known parameters (temperature, speed, etc.) and corresponding mechanical properties (roughness, elongation, tensile strength). The dataset is stored in CSV format.

Source : https://www.kaggle.com/code/mannsingh/ultimaker-data-analysis

## [ 7 ] **How Do You Test & Simulate?**
Testing is done by comparing predicted values with actual measured values from the test dataset. Simulations involve running the model on unseen data and checking the accuracy of predictions.

## [ 8 ] **Interpretation of Results:**
If the model performs as expected but have space for impovements therefore retraining with more data or different features can be consided.

## [ 9 ] **Future Steps & Improvements:**
- Explore more advanced machine learning algorithms like Random Forest, Neural Networks.
- Integrate with real-time printing systems.
- Collect more diverse data to improve model accuracy.
- Update the UI.

# FOLDER STRUCTURE

/MRAC_SOFTWARE_3
├── data/
│   ├── training_data.csv
├── src/
│   ├── mrac_2025_software_3.py        # TRAIN MODEL
│   ├── ROS_NODE.py     # ROS NODE
├── models/
│   ├── roughness_model.pkl  # TRAINED MODEL
│   ├── roughness_scaler.pkl  # SCALER
│   ├── feature_columns.pkl  
├── results/
│   ├──prediction_output.csv
│   ├──all_targets.png 
│   ├──roughness.png  
│   ├──tension_strenght.png
│   ├──elongation.png   
├── README.md              # README 
├── requirements.txt       # Python Packages
└── .gitignore           

## Requirements
scikit-learn==1.4.2        # For training and saving ML models
pandas==2.2.2              # For CSV file manipulation
numpy==1.26.4              # For numerical calculations
matplotlib==3.8.4          # For training and result plots
seaborn==0.13.2            # For visualizations (heatmaps, pair plots)
joblib==1.4.0              # For saving/loading models and scalers


