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
    if target.shape == (2,) or target.shape == (1,2) or target.shape == (2,1):
        eccentricity = numpy.sqrt(numpy.sum((target-position)**2))
        cosalpha = (target - position)[0] / eccentricity
        sinalpha = (target - position)[1] / eccentricity
        _sigma = sdn_level * eccentricity
        sigma = numpy.array([[_sigma, 0], [0, 3*_sigma/4]])
        P = numpy.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
        return P @ sigma @ P.T
    elif target.shape == (1,) or target.shape == (1,1):
        eccentricity = numpy.sqrt(numpy.sum((target-position)**2))
        return numpy.array([sdn_level * eccentricity]).reshape(1,1)
    else:
        raise NotImplementedError
