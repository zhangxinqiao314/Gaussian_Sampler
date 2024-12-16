docker compose up
docker build -t jupyter-notebook .
# sudo docker rm -f $(sudo docker ps -aq --filter ancestor=jupyter-notebook)
docker run -p 8888:8888 -v $(pwd):/workspace jupyter-notebook