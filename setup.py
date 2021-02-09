import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="edgeprediction", 
    version="0.2.1.2",
    author="Dan Bean",
    author_email="daniel.bean@kcl.ac.uk",
    description="Predict missing edges in a knowledge graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KHP-Informatics/ADR-graph",
    packages=['EdgePrediction', 'EdgePrediction.utils'],
    install_requires=[
        'numpy~=1.20.0',
        'python-igraph~=0.8.3',
        'rpy2==3.4.2',
        'scipy~=1.6.0',
        'db-edges==0.0.3',
        'pandas~=1.2.1',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)