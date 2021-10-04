import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='toolbox',
    version='0.1.0',
    author='Karl Hajjar',
    author_email='karl.hajjar@polytechnique.edu',
    description='wide-networks python package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/karl-hajjar/wide-networks.git',
    project_urls={
        "wide-networks": "https://github.com/karl-hajjar/wide-networks.git"
    },
    license='MIT',
    packages=['wide-networks'],
    install_requires=['Click',
                      'clickclick',
                      'flask',
                      'fastapi',
                      'pandas',
                      'numpy',
                      'dateparser',
                      'pyyaml',
                      'scikit-learn',
                      'matplotlib',
                      'seaborn',
                      'pytorch',
                      'torchvision',
                      'pytorch-lightning==0.8.5',
                      'tensorboard==2.2.2',
                      'tensorflow'],
)
