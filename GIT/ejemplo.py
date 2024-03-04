import requests
import pandas as pd

def download_and_eda(url):
    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        # Save the dataset to a file
        with open('dataset.csv', 'wb') as file:
            file.write(response.content)
        print('Dataset downloaded successfully.')
        
        # Perform EDA
        df = pd.read_csv('dataset.csv')
        # Add your EDA code here
        
        # Return the EDA results or perform any other desired actions
        return df.head()
    else:
        print('Failed to download the dataset.')

# Example usage
url = 'https://example.com/dataset.csv'

