import numpy

def eccentric_noise(target, position, sdn_level):
    eccentricity = numpy.sqrt(numpy.sum((target-position)**2))
    cosalpha = (target - position)[0] / eccentricity
    sinalpha = (target - position)[1] / eccentricity
    _sigma = sdn_level * eccentricity
    sigma = numpy.array([[_sigma, 0], [0, 4*_sigma/3]])
    P = numpy.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
    return P @ sigma @ P.T
