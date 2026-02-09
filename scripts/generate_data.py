import numpy as np

from l2d.env.uni_instance_gen import generate_uniform_sjssp_instance

j = 20
m = 10
l = 1
h = 99
batch_size = 100
seed = 200

np.random.seed(seed)

data = np.array([generate_uniform_sjssp_instance(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
print(data.shape)
np.save(f'data/generated/generatedData{j}_{m}_Seed{seed}.npy', data)
