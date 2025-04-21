import numpy as np

# Bots' initial condition
x = np.array([[ -90.36652699,  -90.33972866,  -90.77837917,  -89.63154712 ],
              [ -45.22090462,  -69.31074235,  -57.0937422 ,  -80.53216571 ]])

# Range in Y-axis
y_top = 0.10
y_bottom = -0.10

# Bots' convetered coordinate
x_scalled = x.copy()

for i in range(4):
    print(x[1,0], x[1,-1])
    y_i = (x[1,i] - x[1,0]) * (1/np.abs(x[1,0] - x[1,-1])) * (y_top - y_bottom) + y_top
    x_scalled[0, i] = 0
    x_scalled[1, i] = y_i

print("x_scalled: \n", x_scalled)
