import tkinter
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torch
import math

class Draw:
    def __init__(self, device, model):

        # region: creating root window
        self.root = tkinter.Tk()
        self.root.title("Hand Written Digit Recognition")
        self.root.geometry("400x500")
        # endregion

        # region: creating canvas
        self.canvas = tkinter.Canvas(self.root, width = 280, height = 280, bg = 'black')
        self.canvas.pack(pady = 20)
        # endregion

        # region: creating PIL image for drawing
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        # endregion

        # region: binding mouse event
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        # endregion

        # region: display prediction results
        self.result_label = tkinter.Label(self.root, text = "Prediction Result: -- ", font = ("Arial", 16))
        self.result_label.pack(pady = 10)

        self.confidence_label = tkinter.Label(self.root, text = "Confidence: -- ", font = ("Arial", 12))
        self.confidence_label.pack()
        # endregion

        # region: clear button
        self.clear_button = tkinter.Button(self.root, text = "Clear", command = self.clear_canvas, font = ("Arial", 12))
        self.clear_button.pack(pady = 10)
        # endregion

        # region: record drawing state
        self.drawing = False
        # endregion

        # region: data preprocessing
        # to keep size of data the same as the trained data
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # endregion

        # region: load model
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        # endregion
        
    def start_draw(self, event):
        """start drawing when the mouse button pushed down"""
        self.drawing = True
        self.draw_point(event.x, event.y)
    
    def draw_motion(self, event):
        """"drawing in mouse's motion"""
        if self.drawing:
            self.draw_point(event.x, event.y)
    
    def end_draw(self, event):
        """release the mouse, stop drawing and start prediction"""
        self.drawing = False
        self.predict_digit()
    
    def draw_point(self, x, y):
        """draw point at specific point"""
        radius = 8
        # draw on tkinter canvas
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill = 'white', outline = 'white')
        
        # also draw on PIL image
        self.draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill = 255)
    
    def predict_digit(self):
        """predict hand written digit"""
        try:
            # preprocess image data
            # unsqueeze(0) to eliminate the 'channels' dimension
            img_tensor = self.transform(self.image).unsqueeze(0).to(self.device)

            # predict
            with torch.no_grad():
                output = self.model(img_tensor)
                prediction = torch.argmax(output, dim = 1).item()
                confidence = torch.max(output, dim = 1)[0].item()
                confidence = math.exp(confidence)

            # refresh display
            self.result_label.config(text = f"Prediction Result: {prediction} ")
            self.confidence_label.config(text = f"Confidence: {(100*confidence):.2f}% ")

        except Exception as e:
            print(f"unexpected error occured during prediction: {e}")

    def clear_canvas(self):
        """clear canvas"""
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text = "Prediction Result: -- ")
        self.confidence_label.config(text = "Confidence: -- ")

    def run(self):
        """run GUI"""
        self.root.mainloop()