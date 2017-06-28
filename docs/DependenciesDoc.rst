Dependencies
************

The EdgePrediction library depends on several other Python libraries.
A Docker image is provided with all the necessary dependencies
preconfigured and is the easiest way to start. Alternatively, a full
list is provided below.


Dependencies list
=================

* igraph

* numpy

* scipy

* rpy2

The easiest way to get set up is to use a distribution such as
anaconda, and then install graph and rpy2 following the same steps as
for the Docker image below.


Docker image
============

A docker image is provided here.


Manual - create Docker image
============================

Start docker and connect to default machine:

   docker-machine start default
   eval "$(docker-machine env default)"

pull the anaconda image:

   docker pull continuumio/anaconda

run the anaconda image and note the id:

   docker run -it continuumio/anaconda

In this example, the id is #d60ea25a91b6.

Now we’re working *inside the image shell*. Install build tools:

   apt-get install build-essential

Install igraph with pip:

   pip install -i https://pypi.anaconda.org/pypi/simple python-igraph

Install R and rpy2 with conda:

   conda install -c r rpy2

To commit these changes to the image, first exit the image shell:

   exit

Now commit changes to the base image. Replace YOUR NAME with your
name, in quotes. Replace d60ea25a91b6 with the id you noted down
above. Replace python-image with whatever you want your new image to
be called:

   docker commit -m “installed igraph, r, rpy2" -a “YOUR NAME“ d60ea25a91b6 python-image

You should get a sha256 hash in response. To check the image has been
saved, run:

   docker images

Somewhere in the output you should see your image:

   REPOSITORY             TAG                 IMAGE ID            CREATED             SIZE
   python-image           latest              fd36318d71d9        48 seconds ago      2.345 GB

Now you can run this image as:

   docker run -it python-image
