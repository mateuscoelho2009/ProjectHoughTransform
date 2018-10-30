#!/bin/sh

docker run --rm -p 8888:8888 -p 4040:4040 -p 4041:4041 -v /home/osboxes/Desktop/docker-spark:/home/jovyan/work --name spark jupyter/pyspark-notebook
