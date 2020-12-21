from distutils.core import setup

setup(
    name='dlop_ldn_function_bases',
    version='1.0',
    author='Andreas Stöckel',
    author_email='astoecke@uwaterloo.ca',
    description='Functions for generating DLOP and LDN bases.',
    packages=['dlop_ldn_function_bases'],
    license='License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy >= 1.19',
        'scipy >= 1.5',
    ],
)

