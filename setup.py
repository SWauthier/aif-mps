import setuptools

with open("requirements.txt", "r") as req:
    requirements = req.read().splitlines()

setuptools.setup(
    name="mpstwo",
    version="0.1.0",
    author="DML group",
    author_email="samuel.wauthier@ugent.be",
    description="Toolkit for active inference experiments with tensor networks",
    url="https://gitlab.ilabt.imec.be/swauthie/tn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 2 - Pre-Alpha",
    ],
    packages=setuptools.find_packages(),
    python_requires=">3.6",
    install_requires=requirements,
)
