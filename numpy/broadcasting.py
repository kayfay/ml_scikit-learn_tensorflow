import numpy as np

# Arithmetic operators apply elementwise.
a = np.array([14, 23, 32, 41])
b = np.array([5, 4, 3, 2])
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a // b =", a // b)
print("a % b =", a % b)
print("a ** b =", a**b)

# When length is not the same broadcasting rules are used.
# If arrays rank, 1 prepended to smaller rank.
h = np.arange(5).reshape(1, 1, 5)
# array([[[0, 1, 2, 3, 4]]])

#   [0, 1, 2, 3, 4]]
h + [10, 20, 30, 40, 50]

# array([[[10, 21, 32, 43, 54]]])

# If arrays with 1 along a dimension are treated like the largest shape.
k = np.arange(6).reshape(2, 3)

#  array([[0, 1, 2],
#         [3, 4, 5]])

k + [[100], [200]]

#  array([[100, 101, 102],
#         [203, 204, 205]])

# Or with both, prepended and priority.
k + [100, 200, 300]

#  array([[100, 201, 302],
#         [103, 204, 305]])

# Broadcasting's nature for simplicity.
k + 1000

#  array([[1000, 1001, 1002],
#         [1003, 1004, 1005]])

# If arrays don't match and error is generated.
try:
    k + [33, 44]

except ValueError as e:
    print(e)
