.PHONY: install calibrate train tensorboard clean

install:
	pip install -r requirements.txt

calibrate:
	python calibrate.py

train:
	python train.py

tensorboard:
	tensorboard --logdir runs/

clean:
	rm -rf runs/ models/ __pycache__/ *.pyc
