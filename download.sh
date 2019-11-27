# make folder
mkdir -p data/

# album_covers_t:
wget https://archive.org/download/audio-covers/album_covers_t.tar
tar -xvf album_covers_t.tar
rm album_covers_t.tar
mv album_covers_t data/
# imdb faces:
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
tar -xvf imdb_crop.tar
rm imdb_crop.tar
mv imdb_crop data/
# cityscapes needs registration!
# UTKFace needs gdrive api https://susanqq.github.io/UTKFace/
# simpsons-faces needs kaggle api https://www.kaggle.com/kostastokis/simpsons-faces
# edges2shoes:
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz
tar -zxvf edges2shoes.tar.gz
rm edges2shoes.tar.gz
mv edges2shoes data/
