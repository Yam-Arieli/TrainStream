from setuptools import setup, find_packages

setup(
    name="trainstream",
    version="0.2.0",
    description="Streaming coreset training for large-scale datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yam Arieli",
    url="https://github.com/Yam-Arieli/TrainStream",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "scikit-learn",
    ],
    extras_require={
        "scanpy": ["scanpy", "scipy"],
    },
)
