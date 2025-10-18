from setuptools import setup, find_packages

# relative links to absolute
with open("./README.md", "r") as f:
    readme = f.read()
readme = readme.replace('src="./img_phy_sim/raytracing_example.png"', 'src="https://github.com/xXAI-botXx/Image-Physics-Simulation/raw/main/img_phy_sim/raytracing_example.png"')

setup(
    name='img-phy-sim',
    version='0.2',
    packages=['img_phy_sim'],# find_packages(),
    install_requires=[
        # List any dependencies here, e.g. 'numpy', 'requests'
        "numpy",
        "opencv-python",
        "matplotlib",
        "scikit-image"
    ],
    author="Tobia Ippolito",
    description = 'Physical Simulations on Images.',
    long_description = readme,
    long_description_content_type="text/markdown",
    include_package_data=True,  # Ensures files from MANIFEST.in are included
    download_url = 'https://github.com/xXAI-botXx/Image-Physics-Simulation/archive/v_01.tar.gz',
    url="https://github.com/xXAI-botXx/Image-Physics-Simulation",
    project_urls={
        "Documentation": "https://xxai-botxx.github.io/Image-Physics-Simulation/img_phy_sim",
        "Source": "https://github.com/xXAI-botXx/Image-Physics-Simulation"
    },
    keywords = ['Simulation', 'Computer-Vision', 'Physgen'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',      # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
)