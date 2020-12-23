# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:07:16 2020

@author: Ali
"""

from navigation import Navigation


# please adjust the parameters below
env_path = "Banana_Windows_x86_64/Banana.x86_64"

prioritize_er=True
double_dqn=True
drop_out = True


# Create a navigation instance
nav = Navigation(env_path, criteria=13, seed=0, prioritize_er=prioritize_er, double_dqn=double_dqn, drop_out=drop_out)

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