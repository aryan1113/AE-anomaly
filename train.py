from dataloader import LogisticsDataPreprocessor
from utils import train_anomaly_detection_autoencoder

def main():
    preprocessor = LogisticsDataPreprocessor('/kaggle/input/logistics-and-supply-chain-dataset/dynamic_supply_chain_logistics_dataset.csv')
    train_loader, test_loader, scaler = preprocessor.preprocess()
    
    input_dim = train_loader.dataset.features.shape[1]
    autoencoder, max_error = train_anomaly_detection_autoencoder(train_loader, test_loader, input_dim)
    
    print("Training complete. Best model saved to 'best_autoencoder.pth'")

if __name__ == '__main__':
    main()