from vidstab import VidStab
import matplotlib.pyplot as plt
import numpy as np

stabilizer = VidStab()
stabilizer.gen_transforms("Vid/test1.mov")
np.savetxt("Output/data.txt", stabilizer.transforms, delimiter=",")

# File at TRANSFORMATIONS_PATH is of the form shown below.
# The 3 columns represent delta x, delta y, and delta angle respectively.
