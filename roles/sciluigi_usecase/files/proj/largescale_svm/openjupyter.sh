#!/bin/bash
luigid --background --logdir . --state-path . --port 8082 --address localhost
firefox http://localhost:8082/static/visualiser/index.html &
sleep 5
jupyter notebook wffindcost.ipynb &
sleep 5
gnome-system-monitor &
$SHELL # Keep the terminal open, so that Jupyter does not die
