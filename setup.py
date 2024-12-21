from setuptools import setup, find_packages

setup(
    name="Enrollment Conversion Rate Optimization",  
    version="0.1.0",  # Initial version
    description="A package for analyzing and optimizing lead conversion rates.",
    author="Minal Devikar",
    author_email="meenal.madankar@gmail.com",
    packages=find_packages(),  
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
    ],
    python_requires=">=3.7",  # Specify Python version compatibility
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
