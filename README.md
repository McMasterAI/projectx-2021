# projectx-2021

## Bioconductor

* Use r/bioconductor to fetch RNA-seq data per Ramirez et al.
* It runs in a docker container as configured in `docker-compose.yml`
* Install `docker`, `docker-compose`, and run `docker-compose up` to start the r/bioconductor container
  * Make sure the docker daemon is running
  * Recommend to install Docker Desktop
    * When the Docker Desktop is running, the daemon should be up
* Go to `localhost:8787` to login to RStudio
  * Username is `rstudio`
  * Password is `bioc`
* The `~/projectx-2021` directory of the container maps to `./r` of this project
