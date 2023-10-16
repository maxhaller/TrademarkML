from setuptools import find_packages, setup

setup(
    name="TrademarkML",
    version="1.0",
    description="Predicting the Likelihood of Confusion under Article 8 EUTMR",
    author="Maximilian Haller",
    author_email="maximilian.haller@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "pandas==2.0.2",
        "scikit-learn==1.3.1",
        "jellyfish==1.0.1",
        "spacy==3.7.1",
        "keras==2.14.0",
        "opencv-python==4.8.1.78",
        "tensorflow==2.14.0",
        "Pillow==10.0.1",
        "python-Levenshtein==0.23.0",
        "strsimpy==0.2.1",
        "matplotlib==3.8.0"
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)