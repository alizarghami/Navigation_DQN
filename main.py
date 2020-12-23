# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:07:16 2020

@author: Ali
"""

from navigation import Navigation


# please adjust the parameters below
env_path = "C:/Users/Ali/Documents/udacity/projects/environments/Banana_Windows_x86_64/Banana.x86_64"

# Create a navigation instance
nav = Navigation(env_path, criteria=13, seed=0)

# Train the model
outcome = nav.train()
# Save the trained model if the criteria is reached
if outcome:
    nav.save_model()

# Load the pre-trained model
nav.load_model()
# Evaluate the model
nav.evaluate()

# Close the unity environment
nav.close_env()