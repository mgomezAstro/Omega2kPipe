from setuptools import setup

description = """
Omega2kpipe is an unofficial pipeline reduction of the near infrared camera Omega2000
mounted in the 3.5m telescope at Calar Alto Observatory. The pipeline is based on
the Omega2000 reduction techniques document that is available in its official website
[here](https://www.caha.es/es/telescope-3-5m-2/omega-2000).
"""

install_requires = [
    "astropy>=6.1.2",
    "ccdproc>=2.4.2",
]

setup(
    name="Omega2kpipe",
    version="1.0.0b",
    packages="Omega2kPipe.src",
    package_dir={"": "src"},
    license="GPL-3.0",
    author="mgomez",
    author_email="mgomez_astro@outlook.com",
    description=description,
    install_requires=install_requires,
)
