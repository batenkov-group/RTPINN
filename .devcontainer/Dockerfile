FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

RUN apt-get update && apt-get install -y \
    build-essential autoconf automake libtool git zsh wget gzip tar make gcc g++ gfortran flex perl \
    libnetcdf-dev libgsl-dev \
    && rm -rf /var/lib/apt/lists/*



SHELL ["/bin/zsh", "-c"]

WORKDIR /tmp

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

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh && \
    bash Miniconda3-latest-Linux-$(uname -m).sh -b -p ${CONDA_DIR} && \
    rm Miniconda3-latest-Linux-$(uname -m).sh
ENV PATH=${CONDA_DIR}/bin:${PATH}


USER vscode
RUN conda init --all && \
    source ~/.zshrc && \
    conda create -n rtpinn python=3.10 -y && \
    conda activate rtpinn && \
    conda install -c pytorch pytorch -y && \
    conda install -c conda-forge numpy pandas matplotlib scipy xarray ipykernel -y && \
    pip install sobol-seq && \
    conda clean -afy

WORKDIR /workspace