from setuptools import setup, find_packages

setup(
    name="VoA",
    version="0.0.1",
    author="Mateusz Przyborowski, Krzysztof Suwada",
    description="Tools proposed in the paper 'Validation of Association'",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://arxiv.org/pdf/1904.06519",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "plotly",
        # Add any other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

