FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

RUN apt-get update && apt-get install -y \
    build-essential autoconf automake libtool git zsh wget gzip tar make gcc g++ gfortran flex perl \
    libnetcdf-dev libgsl-dev \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/zsh", "-c"]

WORKDIR /tmp

# Install libRadtran
RUN wget -q --user=libRadtran --password=libRadtran "http://www.libradtran.org/download/libRadtran-2.0.6.tar.gz" && \
    tar xzf libRadtran-2.0.6.tar.gz --warning=no-unknown-keyword && \
    cd libRadtran-2.0.6 && \
    echo "$(pkg-config --cflags gsl) $(nc-config --cflags)" > /tmp/cflags && \
    echo "$(pkg-config --libs gsl) $(nc-config --libs)" > /tmp/ldflags && \
    export CFLAGS="$(cat /tmp/cflags)" && \
    export CXXFLAGS="$(cat /tmp/cflags)" && \
    export LDFLAGS="$(cat /tmp/ldflags)" && \    
    ./configure && \
    cd libsrc_c && make clean && cd .. && \
    cd src && make clean && cd .. && \
    make && \
    make install && \
    cd .. && \
    rm -rf libRadtran-2.0.6* cflags ldflags

RUN wget -q "http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:optprop_v2.1.tar.gz" -O optprop_v2.1.tar.gz && \
    tar xzf optprop_v2.1.tar.gz && \
    cp -r data/* /usr/local/share/libRadtran/data/ && \
    rm -rf optprop_v2.1.tar.gz data

# Install uv
RUN pip install uv

WORKDIR /workspace
