cvsteer
=======

|Travis| |Appveyor| |License (3-Clause BSD)|

LICENSE
=======

This SW is released under both 3-clause BSD and Boost licenses. Note
that this SW uses OpenCV. Please see relevant license files for that
project.

See:

LICNESE.bsd LICENSE.boost

cvsteer
=======

A concise implementation of separable steerable filters via Freeman and
Adelson, including second derivative of Gaussian and its Hilbert
transform (G2 H2), implemented with the OpenCV C++ API

HOWTO
=====

Install latest Polly toolchains and scripts for cmake (not needed but very easy)
================================================================================

::

    wget https://github.com/ruslo/polly/archive/master.zip
    unzip master.zip
    POLLY_ROOT="`pwd`/polly-master"
    export PATH="${POLLY_ROOT}/bin:${PATH}"

Build
=====

::

    polly.py --install --reconfig --verbose --fwd CVSTEER_BUILD_EXAMPLE=ON

Run
===

::

    _install/default/bin/cvsteer-run --input=/tmp/some_file.png --output=/tmp --gain=1.0

.. |Travis| image:: https://img.shields.io/travis/headupinclouds/cvsteer/master.svg?style=flat-square&label=Linux%20OSX%20Android%20iOS
   :target: https://travis-ci.org/headupinclouds/cvsteer
.. |Appveyor| image:: https://img.shields.io/appveyor/ci/headupinclouds/cvsteer.svg?style=flat-square&label=Windows
   :target: https://ci.appveyor.com/project/headupinclouds/cvsteer/branch/master
.. |License (3-Clause BSD)| image:: https://img.shields.io/badge/license-BSD%203--Clause-brightgreen.svg?style=flat-square
   :target: http://opensource.org/licenses/BSD-3-Clause
