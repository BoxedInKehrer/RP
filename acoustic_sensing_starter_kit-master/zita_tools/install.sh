#! /bin/bash
sudo apt install libfftw3-dev jackd qjackctl libsndfile-dev libjack-jackd2-dev libzita-resampler-dev

sudo sh -c "echo '/usr/local/lib64/' > /etc/ld.so.conf.d/usr_local.conf"
sudo ldconfig

echo "Building jclient"
cd zita-jclient/source
sudo make install
cd ../../

echo "Building resampler"
cd zita-resampler/source
sudo make install
cd ../../

echo "Building convolver"
cd zita-convolver/source
sudo make install
cd ../../

echo "Building audiotools"
cd zita-audiotools
sudo make install
cd ..

echo "Building jacktools"
cd zita-jacktools
sudo make install
cd ..
