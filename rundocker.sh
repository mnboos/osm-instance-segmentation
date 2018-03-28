#!/usr/bin/env bash

nvidia-docker run -d --name deeposm -v ~/dev/osm-instance-segmentation:/osm-instance-segmentation -v ~/dev/zurich:/training-data -v ~/dev/raw-images:/source-images deeposm
