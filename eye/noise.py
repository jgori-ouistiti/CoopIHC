import numpy

def eccentric_noise(target, position, sdn_level):
    """ Eccentric noise definition

    * Compute the distance between the target and the current fixation.
    * Compute the angle between the radial component and the x component
    * Express the diagonal covariance matrix in the radial/tangential frame.
    * Rotate that covariance matrix with the rotation matrix P

    :param target: true target position
    :param position: current fixation
    :param sdn_level: signal dependent noise level

    :return: covariance matrix in the XY axis

    :meta public:
    """
    eccentricity = numpy.sqrt(numpy.sum((target-position)**2))
    cosalpha = (target - position)[0] / eccentricity
    sinalpha = (target - position)[1] / eccentricity
    _sigma = sdn_level * eccentricity
    sigma = numpy.array([[_sigma, 0], [0, 4*_sigma/3]])
    P = numpy.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
    return P @ sigma @ P.T
