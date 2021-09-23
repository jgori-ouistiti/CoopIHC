from core.space import Space
import numpy

import sys
_str = sys.argv[1]

if _str == 'init' or _str == 'all':
    s = Space(  numpy.array([1,2,3], dtype = numpy.int16) )
    d = Space(  [-numpy.ones((2,2), dtype = numpy.float32), numpy.ones((2,2), numpy.float32)]  )
    f = Space(  [numpy.array([[-2,-2],[-1,-1]], dtype = numpy.float32), numpy.ones((2,2), numpy.float32)]  )

if _str == 'contains' or _str == 'all':
    space = Space(  numpy.array([1,2,3], dtype = numpy.int16) )
    x = numpy.array([2], dtype = numpy.int16)
    y = numpy.array([2], dtype = numpy.float32)
    yy = numpy.array([2])
    z = numpy.array([5])

    d = Space(  [-numpy.ones((2,2), dtype = numpy.float32), numpy.ones((2,2), numpy.float32)]  )
    x = numpy.array([[1,1],[1,1]], dtype = numpy.int16)
    y = numpy.array([[1,1],[1,1]], dtype = numpy.float32)
    yy = numpy.array([[1,1],[1,1]])
    yyy = numpy.array([[1.0,1.0],[1.0,1.0]])
    z = numpy.array([[5,1],[1,1]], dtype = numpy.float32)

if _str == 'sample' or _str == 'all':
    f = Space(  [numpy.array([[-2,-2],[-1,-1]], dtype = numpy.float32), numpy.ones((2,2), numpy.float32)]  )
