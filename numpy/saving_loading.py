import numpy as np

# Binary format numpy files saving and loading.
a = np.random.rand(2, 3)

np.save("array", a)

with open("array.npy", "rb") as f:
    print(f.read())  # View binary content.

b"\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }                                                          \n\x9e@rX\xbc\x1c\xd6?\x11\xd8\xd96\x1cB\xe9?X\xaf\xf7X\x82\xd5\xc7?\xa4z\xe1\x92\x94\xae\xd7?\x8a&\x0eR?\x1f\xd8?\x08t/\xcc\xa3\xaf\xdb?"

a_loaded = np.load("array.npy")

# Text format saving and loading.
np.savetxt("array.csv", a, delimiter=",")

#  6.919134262305219885e-01,3.757540155766457834e-01,4.036410273494346335e-01
#  1.217573450159186166e-01,8.778611948444898783e-01,8.571935996594026719e-02

with open("array.csv", "rt") as f:
    print(f.read())  # View text content.

a_loaded = np.loadtxt("array.csv", delimiter=",")
a_loaded

#  [[0.69191343 0.37575402 0.40364103]
#   [0.12175735 0.87786119 0.08571936]]

# Zipped format
b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)

np.savez("arrays", array_a=a, array_b=b)

# Loads a dictionary object.
arrays = np.load("arrays.npz")

# Dictionary object.
arrays.keys()
#  ['array_a', 'array_b']

# View key.
arrays["array_a"]
#  array([[0.40582872, 0.28485938, 0.3553134 ],
#         [0.469179  , 0.53965377, 0.57297966]])
