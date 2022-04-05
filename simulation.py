import numba as numba
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import sys
import time

import psutil

print("physical processors", psutil.cpu_count(logical = False))
print("logical processors", psutil.cpu_count(logical = True))

###########################################################
# global variables
###########################################################
gridSize = 400
probTree = 0.8
probBurning = 0.01
probLightning = 0.001
probImmune = 0.3
von_neumann_neighbourhood = False
moore_neighbourhood = True
fig = plt.figure()
colorMap = colors.ListedColormap(['black', 'green', 'red'])
intervals = 100
frame_num = 5000
###########################################################

@njit(parallel=True)
def generate_grid_n_by_n():
    grid = np.zeros((gridSize, gridSize))
    for i in numba.prange(len(grid)):
        for j in prange(len(grid[i])):
            rand_num = np.random.uniform(0.0, 1.0, 1)
            if rand_num < probTree:
                grid[i][j] = 1  # tree is present
            if rand_num < probBurning:
                grid[i][j] = 2  # tree is burning
    return grid

# references for padding : https://stackoverflow.com/questions/44145948/numpy-padding-array-with-zeros
@njit(parallel=True)
def absorbing_boundary_condition(grid, padlen=1):
    m, n = grid.shape
    padded_grid = np.zeros((m + 2 * padlen, n + 2 * padlen), dtype = grid.dtype)
    padded_grid[padlen:-padlen, padlen:-padlen] = grid
    return padded_grid

@njit(parallel=True)
def spread_fire(grid):
    spread_fire_grid = np.copy(grid)
    for i in numba.prange(len(grid)):
        for j in numba.prange(len(grid[i])):
            rand_num = np.random.uniform(0.0, 1.0, 1)

            if grid[i][j] == 2:
                spread_fire_grid[i][j] = 0

            if grid[i][j] == 1 and rand_num > probImmune:
                if von_neumann_neighbourhood:
                    if i == 0 or i == len(grid) - 1 or j == 0 or j == len(grid[i]) - 1:
                        # for corners
                        von_neumann_neighbour_array = []
                        if i != 0:
                            von_neumann_neighbour_array.append(grid[i - 1][j])  # left neighbor
                        if j != len(grid[i]) - 1:
                            von_neumann_neighbour_array.append(grid[i][j + 1])  # bottom neighbor
                        if i != len(grid) - 1:
                            von_neumann_neighbour_array.append(grid[i + 1][j])  # right neighbor
                        if j != 0:
                            von_neumann_neighbour_array.append(grid[i][j - 1])  # top neighbor
                    else:
                        # add neighbors
                        von_neumann_neighbour_array = [
                            grid[i - 1][j],  # left neighbor
                            grid[i][j + 1],  # bottom neighbor
                            grid[i + 1][j],  # right neighbor
                            grid[i][j - 1]  # top neighbor
                        ]

                    for m in numba.prange(len(von_neumann_neighbour_array)):
                        if von_neumann_neighbour_array[m] == 2 or rand_num < probLightning:
                            spread_fire_grid[i][j] = 2

                if moore_neighbourhood:
                    if i == 0 or i == len(grid) - 1 or j == 0 or j == len(grid[i]) - 1:
                        # corners
                        moore_neighbour_array = []
                        if i != 0:
                            moore_neighbour_array.append(grid[i - 1][j])  # left neighbor
                        if j != len(grid[i]) - 1:
                            moore_neighbour_array.append(grid[i][j + 1])  # bottom neighbor
                        if i != 0 and j != len(grid[i]) - 1:
                            moore_neighbour_array.append(grid[i - 1][j + 1])  # bottom left neighbor
                        if i != len(grid) - 1:
                            moore_neighbour_array.append(grid[i + 1][j])  # right neighbor
                        if i != len(grid) - 1 and j != 0:
                            moore_neighbour_array.append(grid[i + 1][j - 1])  # top right neighbor
                        if j != 0:
                            moore_neighbour_array.append(grid[i][j - 1])  # top neighbor
                        if i != 0 and j != 0:
                            moore_neighbour_array.append(grid[i - 1][j - 1])  # top left neighbor
                        if i != len(grid) - 1 and j != len(grid[i]) - 1:
                            moore_neighbour_array.append(grid[i + 1][j + 1])  # bottom right neighbor
                    else:
                        # add neighbors
                        moore_neighbour_array = [
                            grid[i - 1][j - 1],  # top left neighbor
                            grid[i - 1][j],  # left neighbor
                            grid[i - 1][j + 1],  # bottom left neighbor
                            grid[i][j + 1],  # bottom neighbor
                            grid[i + 1][j + 1],  # bottom right neighbor
                            grid[i + 1][j],  # right neighbor
                            grid[i + 1][j - 1],  # top right neighbor
                            grid[i][j - 1]  # top neighbor
                        ]

                    for m in numba.prange(len(moore_neighbour_array)):
                        if moore_neighbour_array[m] == 2 or rand_num < probLightning:
                            spread_fire_grid[i][j] = 2
    return spread_fire_grid[1:-1, 1:-1]


start = time.time()
grid = generate_grid_n_by_n()
image = plt.imshow(grid, cmap=colorMap, aspect='auto')
np.set_printoptions(threshold=sys.maxsize)


def init():
    image.set_data([], [])
    return image,


def animate(i):
    if i == 50:
        end = time.time()
        print(end - start)
        plt.close("all")
    else:
        animate.X = absorbing_boundary_condition(animate.X)
        animate.X = spread_fire(animate.X)
        image.set_array(animate.X)

        return [image]


animate.X = grid
anim = animation.FuncAnimation(fig, animate, interval=intervals, frames=frame_num)
plt.show(block=False)
plt.pause(intervals)

# function to quit the animation
def press(event):
    if event.key == 'q':
        ani.event_source.stop()
cid = fig.canvas.mpl_connect('key_press_event', press)

print(time.time() - start)
