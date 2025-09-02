from setuptools import setup, find_packages

setup(
    name="DynaSD",
    version="0.1.0",
    description="Dynamic Seizure Detection Package",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "h5py",
    ],
    python_requires=">=3.7",
) 