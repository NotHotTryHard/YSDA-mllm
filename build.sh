#!/bin/bash
set -o xtrace

setup_root() {
    apt-get update -qq
    apt-get install -qq -y \
        python3-pip        \
        python3-tk         \
        wget                \
        ;

    ## Unpinned
    # python3 -m pip install -qq             \
    #     torch>=2.0.0                       \
    #     numpy                              \
    #     matplotlib                         \
    #     scikit-learn                       \
    #     scipy                              \
    #     nltk                               \
    #     subword-nmt                        \
    #     pytest                             \
    #     tqdm                               \
    #     bokeh                              \
    #     sacrebleu                          \
    #     ;

    ## Pinned
    python3 -m pip install -qq             \
        torch>=2.0.0                       \
        numpy                              \
        matplotlib                         \
        scikit-learn                       \
        scipy                              \
        nltk                               \
        subword-nmt                        \
        pytest                             \
        tqdm                               \
        bokeh                              \
        sacrebleu                          \
        ;
}

setup_checker() {
    python3 --version
    python3 -m pip freeze
    python3 -c 'import torch; import numpy; import matplotlib.pyplot; import sklearn; import scipy; import nltk; import subword_nmt; import pytest; import tqdm; import bokeh; import sacrebleu; print("All packages imported successfully")'
}

"$@"