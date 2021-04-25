IP=46.101.198.202
docker build --tag reflexology .
docker save reflexology:latest | gzip > ~/dev/docker-images/reflexology_latest.tar.gz

scp ~/dev/docker-images/reflexology_latest.tar.gz sid@$IP:~/

ssh sid@$IP ./load_and_deploy.sh
