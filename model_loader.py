import os
import gdown

# Define the path where you want the model to be saved relative to your project directory
MODEL_DIR = 'models'  # Define the folder to store the model
FILE_PATH = os.path.join(MODEL_DIR, 'yolov10x_best.pt')  # Modify to store inside 'models' folder
FILE_ID = "15YJAUuHYJQlMm0_rjlC-e_VJPmAvjeiE"  # File ID from the shared link on Google Drive
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

def load_model():
    # Check if the model already exists
    if not os.path.exists(FILE_PATH):
        print("Downloading model from Google Drive...")
        # Ensure the directory exists where the model will be saved
        os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
        try:
            # Download the file using gdown
            gdown.download(FILE_URL, FILE_PATH, quiet=False)
            print(f"Model downloaded successfully at: {FILE_PATH}")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            return None
    else:
        print(f"Model already exists at: {FILE_PATH}")
    
    return FILE_PATH