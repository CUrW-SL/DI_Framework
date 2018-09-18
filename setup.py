from setuptools import setup, find_packages

setup(
    name='di_framework',
    version='0.0.1',
    packages=find_packages(),
    url='http://www.curwsl.org',
    license='MIT License',
    author='thilinamad',
    author_email='madumalt@gamil.com',
    description='Data integration framework with algo_wrapper conformity.',
    install_requires=['data_layer', 'algo_wrapper', 'pandas', 'numpy', 'netCDF4'],
    zip_safe=True
)
