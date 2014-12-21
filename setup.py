from setuptools import setup

setup(
    author='Mateusz Moneta',
    author_email='mateuszmoneta@gmail.com',
    name='speedometer',
    install_requires=[
        'numpy',
        'pybrain',
        'cv2',
        'argh==0.26.1'],
    entry_points={
      'console_scripts': [
          'speedometer = speedometer.main:main'
      ]
    },
    package=['speedometer'],
    version='0.1.0',
    zip_safe=False
)
