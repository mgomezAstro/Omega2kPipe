|Astropy|


Omega2kpipe
###########

Omega2kpipe is an unofficial pipeline reduction of the near infrared camera Omega2000
mounted in the 3.5m telescope at Calar Alto Observatory. The pipeline is based on
the Omega2000 reduction techniques document that is available in its official website
`here <https://www.caha.es/es/telescope-3-5m-2/omega-2000>`_.

This pipeline removes the DARK and FLATS from the scientific image according to the
total exposure time and number of integrations. The pipeline also removes the SKY by
computing a median-normilized SKY frame by using the median-mode value of the stacked
pixels of the scientific images. The normalized sky frame is then multiplyied by the
median value of the scientific image and then subtracted: sci - sky * median(sci).

The pipelie is totally based on astropy reduction tools.

Installation
===========


``pip install git+https://github.com/mgomezAstro/Omega2kPipe.git``


Warranty
===========

Omega2kPipe is provided as it is. No warranty at all.

Usage Example
===========

You must run inside the raw fits files data. Then: 

.. code-block:: python
   from omega2kpipe import Omega2kPipe

   pipe = Omega2kPipe()
   pipe.get_master_dark()
   pipe.get_master_flats()
   pipe.reduce_images()
   pipe.remove_sky()



.. |Astropy| image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: https://www.astropy.org/
    :alt: Powered by Astropy