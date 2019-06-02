from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.16.3",
    "scipy>=1.1.0",
    "torch>=1.0.1",
    "matplotlib>=2.2.3",
    "tqdm>=4.26.0",
    "anndata>=0.6.19",
    "scikit-learn>=0.19.2",
    "h5py>=2.8.0",
    "pandas>=0.23.4",
    "loompy>=2.0.17",
    "jupyter>=1.0.0",
    "ipython>=6.5.0",
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author='Jiankang Xiong',
    author_email='hibearme@163.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    description="Scalable Visualization of Massive Single-Cell Data"
                "Using Neural Networks",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='neuralee',
    name='neuralee',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/HibearME/NeuralEE',
    version='0.1.3',
    zip_safe=False,
)
