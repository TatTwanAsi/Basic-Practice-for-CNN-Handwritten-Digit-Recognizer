# Basic handwritten digit recognizer project based on a CNN.
You can run the main.py file in three modes, using command-line arguments: train, test and app
```python
python main.py [mode]
```
| command-line arguments | description |
|------------------------| ------------|
| **`train`** | train network, save .pth file to **`./weights/CNN_model.pth`**. |
| **`test`** | Load the trained model and evaluate its prediction accuracy on the test set. |
| **`app`** | launch an app using a GUI for interaction. |

# Example Commands:
```python
python main.py train
python main.py test
python main.py app
```
