docker compose up
docker build -t jupyter-notebook .
# sudo docker rm -f $(sudo docker ps -aq --filter ancestor=jupyter-notebook)
docker run -it -v /home/m3learning/Northwestern:/home/m3learning/Northwestern -p 8888:8888 $(pwd):/workspace jupyter-notebook