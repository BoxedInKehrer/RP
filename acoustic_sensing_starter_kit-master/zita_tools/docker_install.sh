sh -c "echo '/usr/local/lib64/' > /etc/ld.so.conf.d/usr_local.conf"
ldconfig

echo "Building jclient"
cd zita-jclient/source
make install
cd ../../

echo "Building resampler"
cd zita-resampler/source
make install
cd ../../

echo "Building convolver"
cd zita-convolver/source
make install
cd ../../

echo "Building audiotools"
cd zita-audiotools
make install
cd ..

echo "Building jacktools"
cd zita-jacktools
make install
cd ..
