from setuptools import setup, find_packages

setup(
    name='ultrasound_processing_package',
    version='0.1.0',
    packages=find_packages(),
    author='A Te Neved',
    author_email='te@email.com',
    description='Ultrasound image processing package for noise removal and curved-to-flat transformation.',
    install_requires=[
        'numpy>=1.20',
        'opencv-python',
        'scipy',
        'matplotlib',
        'Pillow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
