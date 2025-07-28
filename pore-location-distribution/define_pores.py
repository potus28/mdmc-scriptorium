import numpy as np
from numba import njit

@njit
def get_distance_mic(ri, rj, hmat):
    dr = get_distance_vector_mic(ri, rj, hmat)
    drmag = np.linalg.norm(dr)
    return drmag

@njit
def get_distance_vector_mic(ri, rj, hmat):
    """From vector FROM i TO j"""
    hmatinv = np.linalg.inv(hmat)
    si = hmatinv @ ri
    sj = hmatinv @ rj
    ds = sj - si
    ds -= np.rint(ds)
    dr = hmat @ ds
    return dr

@njit
def get_unit_vector(v):
    return v / np.linalg.norm(v)

@njit
def find_rmin(points, r, hmat):
    rmin = np.infty
    for rc in points:
        dr = get_distance_mic(rc, r, hmat)
        if dr < rmin:
            rmin = dr
    return rmin


class CylindricalPore():
    def __init__(self, center = np.array([0., 0., 0.]), radius = 5.0, length = 10.0, direction = np.array([0., 0., 1.0]), ncenters = 1000):

        self.center = center
        self.radius = radius
        self.direction = get_unit_vector(direction)

        self.end_point_a = center - 0.5*length * direction
        self.end_point_b = center + 0.5*length * direction

        self.centerline = np.zeros((ncenters,3))

        self.stepsize = length / (ncenters -1)

        for i in range(ncenters):
            self.centerline[i] = self.end_point_a + i * self.stepsize * self.direction

    def is_in_pore(self, r, hmat):
        '''
        rmin = np.infty
        for rc in self.centerline:
            dr = get_distance_mic(rc, r, hmat)
            if dr < rmin:
                rmin = dr
        '''
        #rmin = find_rmin(self.centerline, r, hmat)

        dr = get_distance_vector_mic(self.center, r, hmat)
        drmag = np.linalg.norm(dr)
        adotc = np.dot(self.direction, get_unit_vector(dr))
        theta = np.arccos(adotc)
        rmin = drmag*np.sin(theta)

        if rmin < self.radius:
            return True
        else:
            return False



# BEA zeolite lattice lengths for a 2x2x1 supercell
a = 25.32278
b = 25.32278
c = 26.40612

# Looking down the 100 vector with y pointing right and z pointing up
pore_100_lower_left = CylindricalPore(
        center = np.array([0.5*a, 2.09661, 0.05113]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 1000,
        )

pore_100_lower_right = CylindricalPore(
        center = np.array([0.5*a, 14.758, 0.05113]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 1000
        )

pore_100_upper_left = CylindricalPore(
        center = np.array([0.5*a, 10.56475, 13.15175]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 1000
        )

pore_100_upper_right = CylindricalPore(
        center = np.array([0.5*a, 23.22614, 13.15175]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 1000
        )

# Looking down the 010 vector with z pointing right and x pointing up

pore_010_lower_left = CylindricalPore(
        center = np.array([10.56475, 0.5*b, 6.653]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 1000
        )

pore_010_upper_left = CylindricalPore(
        center = np.array([23.22614, 0.5*b, 6.653]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 1000
        )


pore_010_lower_right = CylindricalPore(
        center = np.array([2.09661, 0.5*b, 19.75325]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 1000
)



pore_010_upper_right = CylindricalPore(
        center = np.array([14.758, 0.5*b, 19.75325]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 1000
        )


