Travis: |Build Status|

Appveyor: |Build status|

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

    build.py --install --reconfig --verbose

Run
===

::

    _install/default/bin/cvsteer_test --demo

TODO
====

Steerable pyramids via Simoncelli http://www.cns.nyu.edu/~eero/steerpyr/

.. |Build Status| image:: https://travis-ci.org/headupinclouds/cvsteer.svg?branch=master
   :target: https://travis-ci.org/headupinclouds/cvsteer
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/64yofu0e25och6tp/branch/master?svg=true
   :target: https://ci.appveyor.com/project/headupinclouds/cvsteer/branch/master
