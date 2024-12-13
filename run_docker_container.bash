sudo docker build -t jupyter-notebook .
sudo docker rm -f $(sudo docker ps -aq --filter ancestor=jupyter-notebook)
sudo docker run -p 8888:8888 -v $(pwd):/workspace jupyter-notebook