from setuptools import setup, find_packages

setup(
    name="trainstream",  # <--- RENAMED
    version="0.1.0",
    description="A streaming coreset framework for training on massive datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/trainstream",
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
    ],
)