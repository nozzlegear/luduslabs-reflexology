# REFlexology
This repo contains the code for reading and analysing REFlex data.

You need to first build the Docker image. 

`docker build --tag reflexology .``

After building, you can run the container locally

`docker run -v local_data_dir:/data` -p 8183:8183 reflexology:latest`

`local_data_dir` refers to the directory on the host machine where the data
will be saved.
