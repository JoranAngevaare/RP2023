import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='RP2023',
                 version='0.1.1',
                 description='Research practicum 2023',
                 author='Joran Angevaare',
                 url='https://github.com/jorana/RP2023',
                 long_description=readme + '\n\n' + history,
                 long_description_content_type="text/markdown",
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + [
                     'pytest',
                     'hypothesis',
                     'boltons'],
                 python_requires=">=3.6",
                 packages=setuptools.find_packages(),
                 scripts=[],
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: BSD License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Programming Language :: Python :: 3.10',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Topic :: Scientific/Engineering :: Physics',
                 ],
                 zip_safe=False)
