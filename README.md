
# Solving Navigation problem from Unity Environments using Deep Reinforcement Learning

### Introduction

This repository contains code for training an intelligent agent that can act efficiently in Navigation unity environment using Deep Reinforcement Learning.

### About the environment

This environment lets the agent roam across a room full of yellow and blue Bananas. The goal is to collect (touch) yellow bananas and avoid the blue ones. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. At each state the agent has 4 possible actions.
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Our goal is to reach score of +13 over 100 consecutive episodes.

### Getting Started

In order to run this code you need to have Python 3 and Jupyter-notebook installed. In addition you need to install the following modules.
* Pytorch: [click here](https://pytorch.org/get-started/locally)
* Numpy: [click here](https://numpy.org/install)
* UnityEnvironment: [click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
* OpenAI Gym: [click here](https://github.com/openai/gym)

You also need to download the Navigation environment from the links below. You need only select the environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
Make sure to decompress the zipped file before running the code

You are strongly suggested to install all the dependencies in a virtual environment. If you are using conda you can create and activate a virtual environment by the following commands:

	```bash
	conda create --name ENVIRONMENT_NAME python=3.6
	conda activate ENVIRONMENT_NAME
	``` 
	
You can deactivate your environment by this command:

	```
	conda deactivate
	```
	
An alternative method for using python virtual environments can be found here: [click here](https://virtualenv.pypa.io/en/latest/)

For more information and instructions on how to install all dependencies check [this link](https://github.com/udacity/deep-reinforcement-learning#dependencieshttps://github.com/udacity/deep-reinforcement-learning#dependencies).

### Instructions

In order to run the code you need to open `banana_navigation.ipynb` in your Jupyter-notebook. Point to the Navigation Environment location on your system where specified in the code and run.
