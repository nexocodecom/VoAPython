from setuptools import setup, find_packages

setup(
    name="VoA",
    version="0.0.5.1",
    author="Mateusz Przyborowski, Krzysztof Suwada, Hubert Bryłkowski",
    description="Tools proposed in the paper 'Validation of Association'",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nexocodecom/VoAPython",
    packages=find_packages(exclude=["webapp", "webapp.*"]),
    install_requires=[
        "numpy >= 1.23.5",
        "numba >= 0.58.1",
        "plotly >= 5.18.0",
        # Add any other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

