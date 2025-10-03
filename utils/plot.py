import matplotlib.pyplot as plt

def plot_training_curves(loss_list):
    plt.plot(range(1, len(loss_list)+1), loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.show()
