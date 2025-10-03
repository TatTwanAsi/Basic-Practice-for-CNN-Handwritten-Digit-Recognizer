import torch
from models.cnn import CNN
from utils.train import train
from utils.test import test
from utils.plot import plot_training_curves as plot
from utils.weight import load_weight, save_weight
from GUI.drawGUI import Draw

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    model = CNN()

    # region: model loading
    load_weight(model, "CNN_model.pth", device)
    # endregion
    
    # region: model training
    # model, loss_list = train(model, device)
    # save_weight(model, "CNN_model.pth")
    # plot(loss_list)
    # endregion

    # region: model prediction accuracy test
    # test(model, device)
    # endregion
    
    # region: GUI for handwritten digit prediction
    draw = Draw(device, model)
    draw.run()
    # endregion

if __name__ == "__main__":
    main()


