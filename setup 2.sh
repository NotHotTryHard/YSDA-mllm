#!/bin/bash
# Creates a conda environment and downloads the required data.
#
# Usage:
#   bash setup.sh          # Full setup (create env + install + download data)
#   bash setup.sh env      # Only create/update conda environment
#   bash setup.sh data     # Only download and prepare data

set -e

ENV_NAME="mllm-hw2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

create_env() {
    echo "=== Setting up conda environment: $ENV_NAME ==="
    
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '$ENV_NAME' already exists."
    else
        echo "Creating new conda environment '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python=3.10 -y

        echo "Installing packages in '$ENV_NAME'..."
        conda run -n "$ENV_NAME" pip install -q \
            torch>=2.0.0 \
            numpy \
            matplotlib \
            scikit-learn \
            scipy \
            nltk \
            subword-nmt \
            pytest \
            tqdm \
            bokeh \
            sacrebleu
        
        echo ""
        echo "=== Conda environment ready ==="
        echo "Activate with: conda activate $ENV_NAME"
    fi
}

download_data() {
    echo "=== Downloading data ==="
    cd "$SCRIPT_DIR"
    
    if [ ! -f data.txt ]; then
        echo "Downloading data.txt..."
        wget -q https://www.dropbox.com/s/yy2zqh34dyhv07i/data.txt?dl=1 -O data/data.txt
        echo "Downloaded data/data.txt"
    else
        echo "data/data.txt already exists, skipping download"
    fi
}

prepare_bpe() {
    echo "=== Preparing BPE tokenized data ==="
    cd "$SCRIPT_DIR"
    
    PYTHON_CMD="${PYTHON:-python3}"
    $PYTHON_CMD - <<'PYEOF'
import os

if os.path.exists("data/train.bpe.ru") and os.path.exists("data/train.bpe.en"):
    print("BPE data already prepared, skipping.")
else:
    print("Preparing BPE tokenization...")
    from nltk.tokenize import WordPunctTokenizer
    from subword_nmt.learn_bpe import learn_bpe
    from subword_nmt.apply_bpe import BPE
    import io

    tokenizer = WordPunctTokenizer()

    def tokenize(x):
        return " ".join(tokenizer.tokenize(x.lower()))

    with open("data/data.txt") as f:
        lines = f.read().split("\n")

    pairs = [line.split("\t")[:2] for line in lines if "\t" in line and len(line.split("\t")) >= 2]
    data_inp = [tokenize(p[1]) for p in pairs]
    data_out = [tokenize(p[0]) for p in pairs]

    # Learn BPE
    bpe_rules = io.StringIO()
    learn_bpe(io.StringIO("\n".join(data_inp + data_out)), bpe_rules, num_symbols=10000)
    bpe_rules.seek(0)
    bpe = BPE(bpe_rules)

    data_inp_bpe = [bpe.process_line(line) for line in data_inp]
    data_out_bpe = [bpe.process_line(line) for line in data_out]

    with open("data/train.bpe.ru", "w") as f:
        f.write("\n".join(data_inp_bpe))
    with open("data/train.bpe.en", "w") as f:
        f.write("\n".join(data_out_bpe))
    print("BPE data prepared: train.bpe.ru, train.bpe.en")
PYEOF
}

case "${1:-all}" in
    env)
        create_env
        ;;
    data)
        download_data
        prepare_bpe
        ;;
    all|"")
        create_env
        download_data
        prepare_bpe
        echo ""
        echo "=== Setup complete ==="
        echo "To activate the environment: conda activate $ENV_NAME"
        echo "To run tests: ./run.py unittest vocab"
        ;;
    *)
        echo "Usage: bash setup.sh [env|data|all]"
        echo "  env  - Create/update conda environment only"
        echo "  data - Download and prepare data only"
        echo "  all  - Full setup (default)"
        exit 1
        ;;
esac
