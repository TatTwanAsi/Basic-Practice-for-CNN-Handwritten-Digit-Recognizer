import torch
import sys
from nets.cnn import CNN
from utils.train import train
from utils.test import test
from utils.plot import plot_training_curves as plot
from utils.weight import load_weight, save_weight
from GUI.drawGUI import Draw

def main():
    if len(sys.argv) != 2:
        print("Instruction: python main.py [train|test|app]")
        return
    mode = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    model = CNN()
    
    # region: model training
    if mode == "train":
        model, loss_list = train(model, device)
        save_weight(model, "CNN_model.pth")
        plot(loss_list)
    # endregion

    # region: model prediction accuracy test
    elif mode == "test":
        load_weight(model, "CNN_model.pth", device)
        test(model, device)
    # endregion
    
    # region: GUI for handwritten digit prediction
    elif mode == "app":
        load_weight(model, "CNN_model.pth", device)
        draw = Draw(device, model)
        draw.run()
    # endregion

    else:
        print(f"unknwon mode: {mode}")
        print("available modes: train, test, app")

if __name__ == "__main__":
    main()


