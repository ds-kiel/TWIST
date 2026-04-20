from ultralytics import YOLO
import os
import wandb
from dotenv import load_dotenv
import os
from ultralytics.utils import SETTINGS


# Load environment variables from .env
load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key  
print("WANDB_API_KEY loaded:", wandb_api_key)

# Initialize Weights & Biases environment
wandb.login(key=wandb_api_key)
SETTINGS["wandb"] = True


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = YOLO('Add model path here')

results = model.train(
    data='add data.yaml file path here',
    epochs=600, 
    imgsz=640,
    patience=100,
    resume=True,
    device=['0'],
    project="project_name",
    # single_cls=False,         #Use for single class benchmarking experiments
    name="run_name",
    )

wandb.finish()