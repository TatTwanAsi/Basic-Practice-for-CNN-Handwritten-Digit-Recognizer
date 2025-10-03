import torch
import os

def load_weight(model, weight_file_name, device):
    try:
        model.load_state_dict(torch.load(f"./weights/{weight_file_name}", map_location = device))
    except FileNotFoundError as e:
        print(f"weight file not found: {e}")
    except RuntimeError as e:
        print(f"model loading failed: {e}")
    else:
        print("model loading succeed!")

def save_weight(model, weight_file_name):
    try:
        weights_dir = "./weights"
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        torch.save(model.state_dict(), f"{weights_dir}/{weight_file_name}")

    except PermissionError as e:
        print(f"permission denied: {e}")
    except OSError as e:
        print(f"disk error: {e}")
    except Exception as e:
        print(f"unexpected error while saving weight file: {e}")