import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class LogisticsDataset(Dataset):
    def __init__(self, features, transform=None):
        self.features = torch.FloatTensor(features)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = self.features[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class LogisticsDataPreprocessor:
    def __init__(self, csv_path, target_columns=None):
        """
        Initialize preprocessor for logistics dataset
        
        Args:
            csv_path (str): Path to the CSV file
            target_columns (list): Columns to use as features
            anomaly_detection (bool): Whether preprocessing is for anomaly detection
        """
        self.df = pd.read_csv(csv_path)
        
        if target_columns is None:
            target_columns = [
                'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate', 
                'eta_variation_hours', 'weather_condition_severity','iot_temperature','route_risk_level']
        
        self.target_columns = target_columns
        self.anomaly_detection = anomaly_detection
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.anomaly_label_scaler = MinMaxScaler()
    
    def preprocess(self, test_size=0.2, random_state=42):
        """
        Preprocess the dataset
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
        
        Returns:
            Train and test dataloaders
        """
        X = self.df[self.target_columns]
        
        X_scaled = self.feature_scaler.fit_transform(X)
        

        X_train, X_test = train_test_split(
                X_scaled, 
                test_size=test_size, 
                random_state=random_state)
        
        train_dataset = LogisticsDataset(X_train)
        test_dataset = LogisticsDataset(X_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=2
        )
        
        return train_loader, test_loader, self.feature_scaler
    
    def inverse_transform_features(self, scaled_features):
        """
        Inverse transform scaled features back to original scale
        
        Args:
            scaled_features (np.array or torch.Tensor): Scaled features
        
        Returns:
            Inverse transformed features
        """
        if torch.is_tensor(scaled_features):
            scaled_features = scaled_features.numpy()
        
        return self.feature_scaler.inverse_transform(scaled_features)
