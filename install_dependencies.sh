mkdir /root/utils
cd /root/utils
wget https://download.redis.io/releases/redis-4.0.8.tar.gz
tar xzf redis-4.0.8.tar.gz
cd redis-4.0.8
make

wget https://github.com/linux-test-project/lcov/releases/download/v1.15/lcov-1.15.tar.gz
tar -xvf lcov-1.15.tar.gz lcov-1.15
cd lcov-1.15
make install -j4
