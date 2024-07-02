from manim import *
import torch
from torch import nn

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CreateNeuralNetwork(Scene):
    def construct(self):
        title = Text("Building a Neural Network").scale(1.5)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Define the neural network layers
        input_layer = Text("Input Layer: 28x28").shift(UP * 3)
        hidden_layer1 = Text("Hidden Layer 1: 512 neurons").shift(UP * 1.5)
        hidden_layer2 = Text("Hidden Layer 2: 512 neurons").shift(DOWN * 0)
        output_layer = Text("Output Layer: 10 classes").shift(DOWN * 1.5)

        self.play(Write(input_layer))
        self.wait(1)
        self.play(Write(hidden_layer1))
        self.wait(1)
        self.play(Write(hidden_layer2))
        self.wait(1)
        self.play(Write(output_layer))
        self.wait(2)

        # Simulate data flow through the network
        input_data = Rectangle(height=0.5, width=0.5, color=BLUE).shift(UP * 3 + LEFT * 3)
        self.play(FadeIn(input_data))
        self.wait(1)

        # Move data to first hidden layer
        self.play(input_data.animate.shift(DOWN * 1.5 + RIGHT * 3))
        self.play(input_data.animate.shift(DOWN * 1.5 + RIGHT * 3))
        self.wait(1)

        # Move data to second hidden layer
        self.play(input_data.animate.shift(DOWN * 1.5))
        self.wait(1)

        # Move data to output layer
        self.play(input_data.animate.shift(DOWN * 1.5))
        self.wait(1)

        # Simulate output
        output_data = Rectangle(height=0.5, width=0.5, color=GREEN).shift(DOWN * 3 + RIGHT * 3)
        self.play(Transform(input_data, output_data))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(input_layer), FadeOut(hidden_layer1), FadeOut(hidden_layer2), FadeOut(output_layer), FadeOut(output_data))
        self.wait(2)