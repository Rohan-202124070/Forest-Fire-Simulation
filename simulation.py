import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import sys

###########################################################
# global variables
###########################################################
gridSize = 800
probTree = 0.8
probBurning = 0.01
probLightning = 0.001
probImmune = 0.3
vonNeumannNeighbourhood = True
mooreNeighbourhood = False
fig = plt.figure()
colorMap = colors.ListedColormap(['black', 'green', 'red'])
###########################################################

def generateGridNByN():
    grid = np.zeros((gridSize, gridSize))
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            randNum = np.random.uniform(0.0, 1.0, 1)
            if randNum < probTree:
                grid[i][j] = 1  # tree is present
            if randNum < probBurning:
                grid[i][j] = 2  # tree is burning
    return grid

def absorbingBoundaryCondition(grid):
    return np.pad(grid, ((1, 1), (1, 1)), 'constant')

def reflectiveBoundaryCondition(grid):
    return np.pad(grid, ((1, 1), (1, 1)), 'constant')

def findVonNeumannNeighbourhood(grid, i, j):
    if i == 0 or i == len(grid) - 1 or j == 0 or j == len(grid[i]) - 1:
        # for corners
        vonNeumannneighbors = []
        if i != 0:
            vonNeumannneighbors.append(grid[i - 1][j])  # top neighbor
        if j != len(grid[i]) - 1:
            vonNeumannneighbors.append(grid[i][j + 1])  # right neighbor
        if i != len(grid) - 1:
            vonNeumannneighbors.append(grid[i + 1][j])  # bottom neighbor
        if j != 0:
            vonNeumannneighbors.append(grid[i][j - 1])  # left neighbor
    else:
        # add neighbors
        vonNeumannneighbors = [
            grid[i - 1][j],  # top neighbor
            grid[i][j + 1],  # right neighbor
            grid[i + 1][j],  # bottom neighbor
            grid[i][j - 1]  # left neighbor
        ]
    return vonNeumannneighbors

def findMooreNeighbourhood(grid, i, j):
    num_neighbor = 1
    index = [i, j]
    left = max(0, index[0] - num_neighbor)         # left neighbor
    right = max(0, index[0] + num_neighbor + 1)    # right neighbor
    bottom = max(0, index[1] - num_neighbor)       # bottom neighbor
    top = max(0, index[1] + num_neighbor + 1)      # top neighbor
    return grid[left:right, bottom:top]

def findNeighbourHasFireRoLightningStrikes(grid, i, j):
    neighbourHasFireRoLightningStrikes = False
    if vonNeumannNeighbourhood:
        vonNeumannNeighbourArray = findVonNeumannNeighbourhood(grid, i, j)
        for m in range(len(vonNeumannNeighbourArray)):
            randNum = np.random.uniform(0.0, 1.0, 1)
            if vonNeumannNeighbourArray[m] == 2 or randNum < probLightning:
                neighbourHasFireRoLightningStrikes = True
                return neighbourHasFireRoLightningStrikes
    if mooreNeighbourhood:
        mooreNeighbourArray = findMooreNeighbourhood(grid, i, j)
        for m in range(len(mooreNeighbourArray)):
            for n in range(len(mooreNeighbourArray[m])):
                randNum = np.random.uniform(0.0, 1.0, 1)
                if mooreNeighbourArray[m][n] == 2 or randNum < probLightning:
                    neighbourHasFireRoLightningStrikes = True
                    return neighbourHasFireRoLightningStrikes
    return neighbourHasFireRoLightningStrikes

def spreadFire(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            randNum = np.random.uniform(0.0, 1.0, 1)

            if grid[i][j] == 2:
                grid[i][j] = 0

            if grid[i][j] == 1 and randNum > probImmune:
                    boolVal = findNeighbourHasFireRoLightningStrikes(grid, i, j)
                    if boolVal:
                        grid[i][j] = 2
    return grid

grid = generateGridNByN()
grid = absorbingBoundaryCondition(grid)
image = plt.imshow(grid, cmap=colorMap, aspect='auto')
np.set_printoptions(threshold=sys.maxsize)
print(grid)

def init():
    image.set_data([], [])
    return image,

def animate(i):
    animate.X = spreadFire(animate.X)
    image.set_array(animate.X)
    return [image]

animate.X = grid
anim = animation.FuncAnimation(fig, animate, interval=5, frames=200)
plt.show()
