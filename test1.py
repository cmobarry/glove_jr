"""
Example usage:

https://github.com/JonathanRaiman/glove

git clone git@github.com:JonathanRaiman/glove.git;
cd glove;


conda deactivate glove1; conda env remove -y -n glove1; 
conda create -y -n glove1 python=3.6.8 scipy=1.3.1 numpy=1.16.4 python-snappy=0.5.4 gxx_linux-64=7.3.0 cython=0.29.13;
conda activate glove1;

python -i test1.py

>>> model.__dict__.keys()
dict_keys(['alpha', 'x_max', 'd', 'cooccurence', 'seed', 'W', 'ContextW', 'b', 'ContextB', 'gradsqW', 'gradsqContextW', 'gradsqb', 'gradsqContextB'])

"""

import os
import numpy as np
import glove

cooccur = {
        0: {
                0: 1.0,
                2: 3.5
        },
        1: {
                2: 0.5
        },
        2: {
                0: 3.5,
                1: 0.5,
                2: 1.2
        }
}

model = glove.Glove(cooccur, d=50, alpha=0.75, x_max=100.0)

print('Before training')
print(f"model.W.shape={model.W.shape}")
print(repr([va.dot(vb)/np.sqrt(va.dot(va) * vb.dot(vb)) for va in model.W for vb in model.W]))
if True:
    for epoch in range(100):
        err = model.train(workers=os.cpu_count(), batch_size=200)
        # print("epoch %d, error %.3f" % (epoch, err), flush=True)

print(f'After training epoch={epoch} err={err}')
print(f"model.W.shape={model.W.shape}")
print(repr([va.dot(vb)/np.sqrt(va.dot(va) * vb.dot(vb)) for va in model.W for vb in model.W]))

