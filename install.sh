mkdir data
cd data

# detection model - feature extraction
mkdir detection
cd detection
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth &
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml &
cd ..

# vilbert model
mkdir vilbert
cd vilbert
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin &
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin &
cd ../


# download flickr8k for testing
mkdir raw
cd raw
mkdir flickr8k
cd flickr8k
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip Flickr8k_text.zip
cd ../../../



#######################################
# install feature extraction tools
# https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark/-/blob/master/INSTALL.md
#######################################

conda install ffmpeg -y
conda install ipython -y
pip install ninja yacs cython matplotlib

mkdir other_projects
cd other_projects


# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../../


# install PyTorch Detection
# git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
# cd maskrcnn-benchmark
# # the following will install the lib with
# # symbolic links, so that you can modify
# # the files if you want and won't need to
# # re-build it
# python setup.py build develop
# cd ../

# install PyTorch Detection - use differenct repo for backward compatibility issue
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark/
python setup.py build develop
cd ../


# install APEX
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../



# install VilBERT
# git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
# cd vilbert-multi-task
# python setup.py develop
# cd ../../











