import rospy
from std_msgs.msg import String
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class NODE:
    def __init__(self):
        # Load model, scaler, and features from pre-trained files
        self.model = joblib.load("/path/to/roughness_model.pkl")
        self.scaler = joblib.load("/path/to/roughness_scaler.pkl")
        self.features = joblib.load("/path/to/feature_columns.pkl")
        
        # Define ROS publisher and subscriber
        self.pub = rospy.Publisher('prediction_output', String, queue_size=10)
        self.sub = rospy.Subscriber('input_data', String, self.callback)
    
    def callback(self, data):
        # Parse the received data
        input_data = self.parse_input(data.data)
        
        # Prepare the data for prediction
        df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=["material", "infill_pattern"])
        
        # Fill missing columns and scale the input
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.features]
        
        scaled_input = self.scaler.transform(df)
        
        # Predict using the trained model
        prediction = self.model.predict(scaled_input)[0]
        
        # Prepare the result to be published
        result = f"Predictions: Roughness={prediction[0]}, Elongation={prediction[1]}, Tensile Strength={prediction[2]}"
        
        # Publish the result
        self.pub.publish(result)
    
    def parse_input(self, input_data_str):
        # Function to parse the input data from the incoming message
        input_values = input_data_str.split(',')
        input_dict = {
            "temperature": float(input_values[0]),
            "speed": float(input_values[1]),
            "layer_height": float(input_values[2]),
            "infill_density": float(input_values[3]),
            "material": input_values[4],
            "infill_pattern": input_values[5]
        }
        return input_dict

if __name__ == '__main__':
    rospy.init_node('ml_rod_node')
    ml_rod_node = NODE()
    rospy.spin()
