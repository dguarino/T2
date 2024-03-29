Analysis instructions
========================

Installing the software stack, and then building and running the model requires a certain amount of time.     
For those interested in evaluating and further analyzing the results of the model, we created a simpler virtual environment on mybinder.org where all the data are available and jupyter notebooks are used to perform the analyses and visualization of the major figures in the paper.

**Coming ...**

Installation instructions
========================

NOTE: The instructions below comply with the version of mozaik at the time of the development of this model (2015-2018). There has been advancements in the mozaik framework that are not supported by the code in this repository.

Dependencies
------------
* python 2.7
* scipy/numpy
* nest (latest release, compiled with mpi)
* mpi4py
* pyNN 
* imagen
* param
* parameters
* quantities 
* neo

Installation
------------

Instructions::

  git clone https://github.com/antolikjan/mozaik.git
  cd mozaik
  python setup.py install
  
Please see below:
 * the installation of the dependencies.
 * the installation on Ubuntu Linux
 * how to run the examples
 
.. _ref-detailed:

Detailed instructions
---------------------

.. _ref-virtual-env:

Virtual env
___________

We recommend to install mozaik using the virtualenv python environment manager (http://pypi.python.org/pypi/virtualenv/) , to prevent potential
conflicts with standard versions of required libraries. Users can follow for example http://simononsoftware.com/virtualenv-tutorial tutorial or just do the following steps:
 
 * Install virtualenv
 * Create (for example in your home directory) a directory where all virtual environments will be created home/virt_env
 * Create the virtual environment for mozaik:: 
    
    virtualenv virt_env/virt_env_mozaik/ --verbose --no-site-packages

 * Load the virtual environment for mozaik by::
 
    source virt_env/virt_env_mozaik/bin/activate

Your shell should look now something like::

(virt_env_mozaik)Username@Machinename:~$

Dependencies 
____________

Note that if the installation is done in your virtualenv environment, it doesn't require any root privilege. Unless specified otherwise
all dependencies should be installed via pip.

 * scipy
 * numpy
 * mpi4py
 * matplotlib (1.1 and higher)
 * quantities
 * imagen
 * PyNN
 * Neo
 * parameters
 * param

For mozaik itself, you need to clone with the help of git::

  git clone https://github.com/antolikjan/mozaik.git
  python setup.py install


VIRTUALENV NOTE: You might already have some of the above packages
if you've used the option --system-site-packages when creating the virtual environment for mozaik.
You can list the packages you have e.g. with the help of yolk):
If you've set up the virt_env with the option --system-site-packages and
you're using scipy, numpy, matplotlib anyway you don't have to install those in your virt_env.

.. _ref-ubuntu:

Ubuntu
------

Following these instruction should give you a working copy of mozaik on a 
fresh installation of Ubuntu (at the time of the writing the version was 16.04)

First the list of ubuntu package dependencies::

  sudo apt-get install python2.7 python-dev python-pip python-nose subversion git libopenmpi-dev g++ libjpeg8 libjpeg8-dev libfreetype6 libfreetype6-dev zlib1g-dev libpng++-dev libncurses5 libncurses5-dev libreadline-dev liblapack-dev libblas-dev gfortran libgsl0-dev openmpi-bin python-tk cmake


Virtual env
____________

Then python virtualenv and virtualenvwrapper (an handy way to manage python virtual environments)::

$ sudo pip install virtualenv
$ sudo pip install virtualenvwrapper

To setup `virtualenvwrapper <http://virtualenvwrapper.readthedocs.org/en/latest//>`_ add the following lines at the top of ~/.bashrc ::

    # virtualenvwrapper
    export WORKON_HOME=~/virt_env
    source /usr/local/bin/virtualenvwrapper.sh
    export PIP_VIRTUALENV_BASE=$WORKON_HOME
    export PIP_RESPECT_VIRTUALENV=true

For the first time, run .bashrc (the next times it will be loaded by your terminal)::      

$ source .bashrc

To create a new managed virtualenv you just need to::

    $ mkvirtualenv --no-site-packages mozaik
    $ workon mozaik
    (mozaik)$>
 

Dependencies 
____________

 
Now you can install in this protected environment all other dependencies::

  pip install --upgrade distribute
  pip install numpy scipy mpi4py matplotlib quantities lazyarray interval Pillow imagen param==1.5.1 parameters neo cython pynn

Now we can install *Nest* (always in the virtual environment):

    - download the latest version from their `website <http://www.nest-initiative.org/index.php/Software:Download>`_
    - untar and cd into it::

        tar xvfz nest-2.12.0.tar.gz
        cd nest-2.12.0
    - then configure, choose if you want mpi. ::
    
       (mozaik)$ cmake -Dwith-mpi=ON -DCMAKE_INSTALL_PREFIX:PATH=$HOME/virt_env/mozaik -Dwith-optimize='-O3' ./
    - finally, by launching make and install, it installs PyNest in the activated virtual environment mozaik::
        (mozaik)$ make
        (mozaik)$ make install
    - Then::
        make installcheck
    - nest will reside in $HOME/virt_env/mozaik/lib/python2.7/site-packages. Check that the package is seen by python using::
        python -c 'import nest'


And, finally, Mozaik::
    
    git clone https://github.com/antolikjan/mozaik.git
    cd mozaik/
    python setup.py install
    
.. _ref-run:

Running the model
----------------
The model has been tested using several stimuli and conditions. A separate run file is used for each:

- `run_luminance_open.py`
- `run_luminance_closed.py`
- `run_contrast_open.py`
- `run_contrast_closed.py`
- `run_temporal_open.py`
- `run_temporal_closed.py`
- `run_spatial_open.py`
- `run_spatial_closed.py`
- `run_size_closed.py`
- `run_size_feedforward.py`
- `run_size_closed_overlapping.py`
- `run_size_closed_nonoverlapping.py`
- `run_orientation_open.py`
- `run_orientation_feedforward.py`
- `run_orientation_closed.py`

Each of these files builds the model in the described condition and provides the corresponding stimuli as described in the paper.    
It can be run using the following type of command line:

mpirun -np 2 python run_<stimulus and condition>.py nest 2 param/defaults '<stimulus and condition folder name>'

