import numpy as np
from scipy.special import jv, hankel1


class SoundHardExact2D(object):
    def __init__(self, t, angular_freq, wavenumber, scat_rad):
        self.t = t
        self.w0 = angular_freq
        self.k = wavenumber
        self.a = scat_rad
        self.f = 0

        self.number_of_terms = np.int(30 + (self.k * self.a)**1.01)

    def incident(self, x):
        k = self.k
        a = self.a

        r = np.sqrt(x[0]**2 + x[1]**2)

        idx = np.where(r < a)

        u_incident = np.exp(1j * k * x[0])
        u_incident[idx] = 0

        return u_incident

    def scatter(self, x):
        k = self.k
        a = self.a

        number_of_terms = self.number_of_terms

        r = np.sqrt(x[0]**2 + x[1]**2)
        t = np.arctan2(x[1], x[0])

        idx = np.where(r < a)

        u_scatter = 0
        for n in range(-number_of_terms, number_of_terms):
            dbessel = jv(n-1, k*a) - n/(k*a) * jv(n, k*a)
            dhankel = n/(k*a)*hankel1(n, k*a) - hankel1(n+1, k*a)
            u_scatter += -(1j)**n * dbessel/dhankel * hankel1(n, k*r) * \
                np.exp(1j*n*t)

        u_scatter[idx] = 0
        return u_scatter

    def total(self, x):
        self.f = self.incident(x) + self.scatter(x)
        return self.f

    def total_time_dependent(self, x):
        result = np.exp(-1j * self.w0 * self.t) * self.f
        return result


class SoundSoftExact2D(object):
    def __init__(self, t, angular_freq, wavenumber, scat_rad):
        self.t = t
        self.w0 = angular_freq
        self.k = wavenumber
        self.a = scat_rad
        self.f = 0

        self.number_of_terms = np.int(30 + (self.k * self.a)**1.01)

    def incident(self, x):
        k = self.k
        a = self.a

        r = np.sqrt(x[0]**2 + x[1]**2)

        idx = np.where(r < a)

        u_incident = np.exp(1j * k * x[0])
        u_incident[idx] = 0

        return u_incident

    def scatter(self, x):
        k = self.k
        a = self.a

        number_of_terms = self.number_of_terms

        r = np.sqrt(x[0]**2 + x[1]**2)
        t = np.arctan2(x[1], x[0])

        idx = np.where(r < a)

        u_scatter = 0
        for n in range(-number_of_terms, number_of_terms):
            u_scatter += -(1j)**n * jv(n, k*a)/hankel1(n, k*a) * \
                hankel1(n, k*r) * np.exp(1j*n*t)

        u_scatter[idx] = 0
        return u_scatter

    def total(self, x):
        self.f = self.incident(x) + self.scatter(x)
        return self.f

    def total_time_dependent(self, x):
        return np.exp(-1j * self.w0 * self.t) * self.f


class PenetrableExact2D(object):
    def __init__(self, t, angular_freq, wavenumber1, wavenumber2, scat_rad):
        self.t = t
        self.w0 = angular_freq
        self.k1 = wavenumber1
        self.k2 = wavenumber2
        self.a = scat_rad
        self.f = 0

        self.number_of_terms = np.max([100, np.int(55 + (wavenumber1 *
                                                         scat_rad)**1.01)])

    def incident(self, x):
        k1 = self.k1
        a = self.a

        r = np.sqrt(x[0]**2 + x[1]**2)

        interior = np.where(r < a)

        u_incident = np.exp(1j * k1 * x[0])
        u_incident[interior] = 0

        return u_incident

    def scatter(self, x):
        k1 = self.k1
        k2 = self.k2
        a = self.a

        number_of_terms = self.number_of_terms

        r = np.sqrt(x[0]**2 + x[1]**2)
        t = np.arctan2(x[1], x[0])

        interior = np.where(r < a)
        exterior = np.where(r >= a)

        u_scatter_interior = 0
        u_scatter_exterior = 0
        for n in range(-number_of_terms, number_of_terms):
            besselk1 = jv(n, k1*a)
            besselk2 = jv(n, k2*a)
            hankelk1 = hankel1(n, k1*a)

            dbesselk1 = jv(n-1, k1*a) - n/(k1*a)*jv(n, k1*a)
            dbesselk2 = jv(n-1, k2*a) - n/(k2*a)*jv(n, k2*a)
            dhankelk1 = n/(k1*a)*hankelk1 - hankel1(n+1, k1*a)

            a_n = (1j**n) * (k2*dbesselk2*besselk1 -
                             k1*dbesselk1*besselk2) / \
                            (k1*dhankelk1*besselk2 -
                             k2*dbesselk2*hankelk1)
            b_n = (a_n*hankelk1 + (1j**n)*besselk1) / besselk2

            u_scatter_exterior += a_n*hankel1(n, k1*r)*np.exp(1j*n*t)
            u_scatter_interior += b_n*jv(n, k2*r)*np.exp(1j*n*t)

        u_scatter_exterior[interior] = 0.0
        u_scatter_interior[exterior] = 0.0

        u_scatter = u_scatter_exterior + u_scatter_interior

        return u_scatter

    def total(self, x):
        self.f = self.incident(x) + self.scatter(x)
        return self.f

    def total_time_dependent(self, x):
        return np.exp(-1j * self.w0 * self.t) * self.f
