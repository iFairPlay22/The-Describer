init: install

install:

	# linux
	apt-get update
	apt-get install wget unzip

	# python
	sudo apt-get install python3-pip
	pip3 install torch torchvision torchaudio nltk pycocotools tqdm spacy
	python3 -m spacy download en
	
	# create a data directory
	mkdir data_dir
	
	# download images and annotations to the data directory
	wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data_dir/
	wget http://images.cocodataset.org/zips/train2014.zip -P ./data_dir/
	wget http://images.cocodataset.org/zips/val2014.zip -P ./data_dir/

		
	# extract zipped images and annotations
	unzip -qq ./data_dir/captions_train-val2014.zip -d ./data_dir/
	unzip -qq ./data_dir/train2014.zip -d ./data_dir/
	unzip -qq ./data_dir/val2014.zip -d ./data_dir/ 

	# remove the zip files
	rm ./data_dir/*.zip
