import numpy as np

import basix


def quadrature_params(type, degree, n):
    if type == "default":
        if n == 1:
            x = [0]
            w = [2]
        elif n == 2:
            x = [-1/np.sqrt(3),
                 1/np.sqrt(3)]
            w = [1,
                 1]
        elif n == 3:
            x = [-np.sqrt(3/5),
                 0,
                 np.sqrt(3/5)]
            w = [5/9,
                 8/9,
                 5/9]
        elif n == 4:
            x = [-np.sqrt(3/7 + 2/7*np.sqrt(6/5)),
                 -np.sqrt(3/7 - 2/7*np.sqrt(6/5)),
                 np.sqrt(3/7 - 2/7*np.sqrt(6/5)),
                 np.sqrt(3/7 + 2/7*np.sqrt(6/5))]
            w = [(18 - np.sqrt(30))/36,
                 (18 + np.sqrt(30))/36,
                 (18 + np.sqrt(30))/36,
                 (18 - np.sqrt(30))/36]
        elif n == 5:
            x = [-1/3*np.sqrt(5 + 2*np.sqrt(10/7)),
                 -1/3*np.sqrt(5 - 2*np.sqrt(10/7)),
                 0,
                 1/3*np.sqrt(5 - 2*np.sqrt(10/7)),
                 1/3*np.sqrt(5 + 2*np.sqrt(10/7))]
            w = [(322-13*np.sqrt(70))/900,
                 (322+13*np.sqrt(70))/900,
                 128/225,
                 (322+13*np.sqrt(70))/900,
                 (322-13*np.sqrt(70))/900]
    elif type == "GLL":
        if n == 2:
            x = [-1,
                 1]
            w = [1,
                 1]
        elif n == 3:
            x = [-1,
                 0,
                 1]
            w = [1/3,
                 4/3,
                 1/3]
        elif n == 4:
            x = [-1,
                 -np.sqrt(1/5),
                 np.sqrt(1/5),
                 1]
            w = [1/6,
                 5/6,
                 5/6,
                 1/6]
        elif n == 5:
            x = [-1,
                 -np.sqrt(3/7),
                 0,
                 np.sqrt(3/7),
                 1]
            w = [1/10,
                 49/90,
                 32/45,
                 49/90,
                 1/10]
        elif n == 6:
            x = [-1,
                 -np.sqrt(1/3+2*np.sqrt(7)/21),
                 -np.sqrt(1/3-2*np.sqrt(7)/21),
                 np.sqrt(1/3-2*np.sqrt(7)/21),
                 np.sqrt(1/3+2*np.sqrt(7)/21),
                 1]
            w = [2/(n*(n-1)),
                 (14-np.sqrt(7))/30,
                 (14+np.sqrt(7))/30,
                 (14+np.sqrt(7))/30,
                 (14-np.sqrt(7))/30,
                 2/(n*(n-1))]
        elif n == 7:
            x = [-1,
                 -np.sqrt(5/11+2/11*np.sqrt(5/3)),
                 -np.sqrt(5/11-2/11*np.sqrt(5/3)),
                 0.0,
                 np.sqrt(5/11-2/11*np.sqrt(5/3)),
                 np.sqrt(5/11+2/11*np.sqrt(5/3)),
                 1]
            w = [2/(n*(n-1)),
                 (124-7*np.sqrt(15))/350,
                 (124+7*np.sqrt(15))/350,
                 256/525,
                 (124+7*np.sqrt(15))/350,
                 (124-7*np.sqrt(15))/350,
                 2/(n*(n-1))]

    pts1d = np.array(x)*0.5+0.5
    wts1d = np.array(w)*0.5

    if degree == 1:
        return pts1d, wts1d

    elif degree == 2:
        pts2d = np.array([[x, y] for y in pts1d for x in pts1d])
        wts2d = np.array([w1*w2 for w1 in wts1d for w2 in wts1d])

        return pts2d, wts2d

    elif degree == 3:
        pts3d = np.array(
            [[x, y, z] for z in pts1d for y in pts1d for x in pts1d])
        wts3d = np.array(
            [w1*w2*w3 for w1 in wts1d for w2 in wts1d for w3 in wts1d])

        return pts3d, wts3d


m = 2

# Test quadrature points and weights (1D)
bpts, bwts = basix.make_quadrature("GLL", basix.CellType.interval, m+1)
pts, wts = quadrature_params("GLL", 1, m+1)

print("Is the 1D points equivalent:", np.allclose(bpts.flatten(), pts))
print("Is the 1D weights equivalent:", np.allclose(bwts.flatten(), wts))

# Test quadrature points and weights (2D)
bpts, bwts = basix.make_quadrature("GLL", basix.CellType.quadrilateral, m+1)
pts, wts = quadrature_params("GLL", 2, m+1)

print("Is the 2D points equivalent:", np.allclose(bpts, pts))
print("Is the 2D weights equivalent:", np.allclose(bwts, wts))

# Test quadrature points and weights (3D)
bpts, bwts = basix.make_quadrature("GLL", basix.CellType.hexahedron, m+1)
pts, wts = quadrature_params("GLL", 3, m+1)

print("Is the 3D points equivalent:", np.allclose(bpts, pts))
print("Is the 3D weights equivalent:", np.allclose(bwts, wts))
