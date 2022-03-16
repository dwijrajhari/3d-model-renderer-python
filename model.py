import math
import numpy as np
import cv2 as cv

class Vector():

    '''Define a Vector class with three coordinates'''

    def __init__(self, x = 0, y = 0, z = 0):
        self.mat = np.array([x, y, z, 1], 'float32')

    # Basic transformations - Scale, Translation and Rotation
    def scaled(self, scale_mat):
        return np.matmul(scale_mat, self.mat)
    def translated(self, translate_mat):
        return self.mat[:-1]+translate_mat
    def rotated(self, rotation_mat):
        return np.matmul(rotation_mat, self.mat)

    # Transform the vector inplace
    def scale(self, scale_mat):
        self.mat = self.scaled(scale_mat)
    def translate(self, translate_mat):
        self.mat[:-1] = self.translated(translate_mat)
    def rotate(self, rotation_mat):
        self.mat = self.rotated(rotation_mat)

    def getMagnitude(self):
        ls = math.sqrt(self[0]**2 + self[1]**2 + self[2]**2)
        return ls

    # Combined Transformation - uses the final transformation matrix derived from getTransformationMarix()
    def transform(self, transfm_mat):
        self.mat = self.transformed(transfm_mat)

    def transformed(self, transfm_mat):
        return np.matmul(transfm_mat, self.mat)

    # Magic methods
    def __str__(self):
        return self.mat[:-1].__str__()
    __repr__ = __str__

    def __getitem__(self, i):
        return self.mat[i]

    # Operator Overloading
    def __add__(self, b):
        return Vector(self[0] + b[0], self[1] + b[1], self[2] + b[2])

    def __sub__(self, b):
        return Vector(self[0] - b[0], self[1] - b[1], self[2] - b[2])

    def __mul__(self, b):
        return Vector(self[0]*b, self[1]*b, self[2]*b)

    def __truediv__(self, b):
        return Vector(self[0]/b, self[1]/b, self[2]/b)

    def __neg__(self):
        return Vector(-self[0], -self[1], -self[2])
   

# ----------------------------------------------------------------------    

class Triangle():

    '''Define a Triangle class from three Vectors'''

    def __init__(self, v1, v2, v3):
        self.v = (v1, v2, v3)

    def getNormal(self):
        l1 = self[1]-self[0]
        l2 = self[2]-self[0]
        cv = cross(l1, l2)
        ls = cv.getMagnitude()
        cv = cv/ls
        return cv

    def getCentroid(self):
        centroid = -(self[0]+self[1]+self[2])/3
        return centroid

    def __getitem__(self, i):
        return self.v[i]

    def __str__(self):
        return "Tr( {}, {}, {} )".format(self[0], self[1], self[2])
    __repr__ = __str__

class Mesh():

    '''Define a Mesh made of Triangles'''

    def __init__(self, triangles):
        self.triangles = triangles

    def sortMesh(self):
        self.triangles.sort(key = lambda x: min(x[i][2] for i in range(3)), reverse=True)
        # self.triangles.sort(key = lambda x: x.getCentroid()[2])

#---------------------------------------------------------------------------
# Useful Constants
PI = math.pi
WHITE = (255, 255, 255)

# -------------------------------------------------------------------------
# Functions to return transformation matrices
def getRotationMatrix(angle, type = 'r', plane = 'xy'):
    if type == 'd':
        angle = math.radians(angle)
    cs = math.cos(angle)
    sn = math.sin(angle)
    if plane == 'xy':
        arr = np.array([
            [cs, -sn, 0, 0],
            [sn, cs, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype = 'float32')
    elif plane == 'yz':
        arr = np.array([
            [1, 0, 0, 0],
            [0, cs, -sn, 0],
            [0, sn, cs, 0],
            [0, 0, 0, 1]
        ], dtype = 'float32')
    elif plane == 'zx':
        arr = np.array([
            [cs, 0, -sn, 0],
            [0, 1, 0, 0],
            [sn, 0, cs, 0],
            [0, 0, 0, 1]
        ], dtype = 'float32')
    return arr

def getScaleMatrix(order):
    return np.array([
        [order[0], 0, 0, 0],
        [0, order[1], 0, 0],
        [0, 0, order[2], 0],
        [0, 0, 0, 1]
    ], 'float32')

def getTransformationMatrix(matrices = (), translate_mat = np.array([0, 0, 0])):
    ret_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], 'float32')
    for i in matrices:
        ret_matrix = np.matmul(i, ret_matrix)
    for i in range(3):
        ret_matrix[i][3]+= translate_mat[i]
    return ret_matrix

def cross(a, b):
    return Vector(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# ------------------------------------------------------------------------
# implementing a .obj file reader
def objFileReader(filename):
    vecDump = []
    triDump = []
    f = open(filename, 'r')
    for line in f.readlines():
        ls = line.strip().split(' ')
        if ls[0] == 'v':
            vecDump.append(Vector(float(ls[1]), float(ls[2]), float(ls[3])))
        elif ls[0] == 'f':
            lsr = list(map(int, ls[1:]))
            triDump.append(Triangle(vecDump[lsr[0]-1], vecDump[lsr[1]-1], vecDump[lsr[2]-1]))
    f.close()
    return vecDump, triDump

#----------------------------------------------------
# Display Properties

# 1. Fundamental
screen_width = 600
screen_height = 600
view_angle = PI/2
Zfar = 10
Znear = 0.1

# 2. Derived
aspect_ratio = screen_width/screen_height
q_value = Zfar/(Zfar-Znear)
tn = 1/math.tan(view_angle/2)

projection_mat = np.array([
    [aspect_ratio*tn, 0, 0, 0],
    [0, -tn, 0, 0],
    [0, 0, q_value, 1],
    [0, 0, -Znear*q_value, 0]
], 'float32')

#------------------------------------------------------------------
#Camera, light sources and the global_VecDump
camera_position = Vector(0, 0, 0)
planeLightSource1 = Vector(1, 0, 0)
planeLightSource2 = Vector(-1, 0, 0)
planeLightSource3 = Vector(0, 0, -1)

global_VecDump = []

#-----------------------------------------------------------------
# Screen Methods
def getProjVec(vect):
    projected_vec = Vector()
    y = np.matmul(vect.mat, projection_mat)[:-1]
    zp = vect[2]
    if zp != 0:
        projected_vec.mat = ((y/zp) + 1) * (screen_height/2)
    else:
        projected_vec.mat = (y + 1) * (screen_height/2)
    projected_vec.mat = projected_vec.mat.astype('int32')
    return projected_vec

def drawLine(a, b, screen):
    cv.line(screen, (a[0], a[1]), (b[0], b[1]), thickness = 1, color = WHITE, lineType = cv.LINE_AA)

#-------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # Some Transformation
    rt_r = getRotationMatrix(3, type = 'd', plane = 'xy')
    rt_l = getRotationMatrix(-3, type = 'd', plane = 'xy')
    tmp = getTransformationMatrix(translate_mat = np.array([0, 0, -6]))
    r_left = getTransformationMatrix((tmp, rt_l), translate_mat = np.array([0, 0, 6]))
    r_right = getTransformationMatrix((tmp, rt_r), translate_mat = np.array([0, 0, 6]))


    # Reading the teapot.obj file
    global_VecDump, teapot = objFileReader('teapot.obj')
    teapotMesh = Mesh(teapot)

    for i in global_VecDump:
        i.translate(np.array([0, -3.5, 6]))

    while True:

        # Create a blank SWxSH BGR image
        grid = np.zeros((screen_width, screen_height, 3), 'uint8')
        
        # Painters' Algorithm - Sorting the Triangles in the Mesh
        teapotMesh.sortMesh()

        # Drawing the Triangles on screen with lighting/shading effects
        for tri in teapotMesh.triangles:
            tri_nl = tri.getNormal()
            if dot(tri_nl, tri.getCentroid())>0:

                # Projecting the 3D vectors on 2D screen
                v0 = getProjVec(tri[0])
                v1 = getProjVec(tri[1])
                v2 = getProjVec(tri[2])

                # Shading Effects
                lumin1 = dot(tri_nl, planeLightSource1)*255
                lumin2 = dot(tri_nl, planeLightSource2)*255
                lumin3 = dot(tri_nl, planeLightSource3)*100

                # Drawing the projected Triangles on screen
                # drawLine(v0, v1, grid)
                # drawLine(v1, v2, grid)
                # drawLine(v2, v0, grid)
                cv.fillPoly(grid, np.array([[[v0[0], v0[1]], [v1[0], v1[1]], [v2[0], v2[1]]]], 'int32'),
                            color = (lumin1, lumin2, lumin3), lineType = cv.LINE_AA)

        # Display and controls
        cv.imshow('Display 0', grid)
        ky = cv.waitKey(0)
        if ky == ord('q'):
            break

        elif ky == ord('d'):
            for i in global_VecDump:
                i.transform(r_right)
        elif ky == ord('a'):
            for i in global_VecDump:
                i.transform(r_left)
        
        # for i in global_VecDump:
        #     i.translate(np.array([0, -0.1, 0]))

    cv.destroyAllWindows()
 

    








