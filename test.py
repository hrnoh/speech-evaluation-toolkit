import pickle
import numpy as np

with open('f0_dist.pkl', 'rb') as f:
    temp = pickle.load(f)

print(len(temp['StarGAN-VC']['m2m']))
print(len(np.concatenate(temp['StarGAN-VC']['m2m'])))