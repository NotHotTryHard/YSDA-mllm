import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import sacrebleu
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vocab import Vocab
from seq2seq import BasicModel, AttentiveModel


def compute_loss(model, inp, out, **flags):
    """
    Compute loss (float32 scalar):
      L = (1/|D|) sum_{X,Y in D} sum_{y_t in Y} -log p(y_t | y_1,...,y_{t-1}, X, theta)

    :param inp: input tokens matrix [batch_size, inp_len]
    :param out: reference tokens matrix [batch_size, out_len]
    :returns: average loss scalar
    """
    mask = model.out_voc.compute_mask(out)  # [batch_size, out_len]

    logits = model(inp, out, **flags)  # [batch_size, out_len, vocab_size]

    batch_size, out_len, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.reshape(batch_size * out_len, vocab_size),
        out.reshape(batch_size * out_len),
        reduction='none',
    ).reshape(batch_size, out_len)  # [batch_size, out_len]

    return (loss * mask).sum() / mask.sum()


def compute_bleu(model, inp_lines, out_lines, bpe_sep="@@ ", **flags):
    """
    Estimates corpora-level BLEU score of model's translations.

    :param model: seq2seq model
    :param inp_lines: list of input tokenized strings
    :param out_lines: list of reference tokenized strings
    :returns: BLEU score (0-100)
    """
    import re
    
    def detokenize(text):
        """Remove BPE separators and fix spacing around punctuation."""
        text = text.replace(bpe_sep, "")
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    with torch.no_grad():
        translations, _ = model.translate_lines(inp_lines, **flags)
        translations = [detokenize(line) for line in translations]
        actual = [detokenize(line) for line in out_lines]

        result = sacrebleu.corpus_bleu(
            translations,
            [actual],
            tokenize="none",
            smooth_method="exp",
        )
    return result.score


def _plot_metrics(metrics, step, plot_interval, save_path=None):
    """Draw one figure with 2 subplots: train loss and dev BLEU. Call every plot_interval steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    steps_loss, losses = zip(*metrics["train_loss"])
    axes[0].plot(steps_loss, losses, alpha=0.7)
    axes[0].set_title("train_loss")
    axes[0].set_xlabel("step")
    axes[0].grid(True)

    if metrics["dev_bleu"]:
        steps_bleu, bleus = zip(*metrics["dev_bleu"])
        axes[1].plot(steps_bleu, bleus, color="blue", alpha=0.7)
    axes[1].set_title("dev_bleu")
    axes[1].set_xlabel("step")
    axes[1].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.close(fig)


def train_model(
    model,
    train_inp,
    train_out,
    dev_inp,
    dev_out,
    inp_voc,
    out_voc,
    device="cpu",
    num_steps=25000,
    eval_interval=100,
    plot_interval=50,
    plot_save_path="outputs/basic_model_metrics.png",
    checkpoint_path="outputs/basic_model_checkpoint.pt",
    **kwargs
):
    """
    Train a seq2seq model with evaluation on dev set.
    """
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    metrics = {"train_loss": [], "dev_bleu": []}
    batch_size = 32

    for step in tqdm(range(1, num_steps + 1)):
        indices = np.random.randint(0, len(train_inp), batch_size)
        inp_batch = inp_voc.to_matrix(train_inp[indices]).to(device)
        out_batch = out_voc.to_matrix(train_out[indices]).to(device)

        optimizer.zero_grad()
        loss = compute_loss(model, inp_batch, out_batch)
        loss.backward()
        optimizer.step()

        metrics["train_loss"].append((step, loss.item()))

        if step % eval_interval == 0:
            with torch.no_grad():
                bleu = compute_bleu(model, dev_inp[:500], dev_out[:500])
            metrics["dev_bleu"].append((step, bleu))

        if plot_interval > 0 and step % plot_interval == 0:
            _plot_metrics(metrics, step, plot_interval, save_path=plot_save_path)

        if step % eval_interval == 0 and np.mean(metrics["dev_bleu"][-10:], axis=0)[1] > 10:
            break

    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path) 

    return metrics


def train_attentive_model(
    model,
    train_inp,
    train_out,
    dev_inp,
    dev_out,
    inp_voc,
    out_voc,
    device="cpu",
    num_steps=25000,
    eval_interval=100,
    plot_interval=50,
    plot_save_path="outputs/attentive_model_metrics.png",
    checkpoint_path="outputs/attentive_model_checkpoint.pt",
    **kwargs
):
    """
    Train an AttentiveModel with evaluation on dev set.
    """
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    metrics = {"train_loss": [], "dev_bleu": []}
    batch_size = 32

    for step in tqdm(range(1, num_steps + 1)):
        indices = np.random.randint(0, len(train_inp), batch_size)
        inp_batch = inp_voc.to_matrix(train_inp[indices]).to(device)
        out_batch = out_voc.to_matrix(train_out[indices]).to(device)

        optimizer.zero_grad()
        loss = compute_loss(model, inp_batch, out_batch)
        loss.backward()
        optimizer.step()
        
        metrics["train_loss"].append((step, loss.item()))

        # your code here /\

        if step % eval_interval == 0:
            with torch.no_grad():
                bleu = compute_bleu(model, dev_inp[:500], dev_out[:500])
            metrics["dev_bleu"].append((step, bleu))

        if plot_interval > 0 and step % plot_interval == 0:
            _plot_metrics(metrics, step, plot_interval, save_path=plot_save_path)

        if step % eval_interval == 0 and len(metrics["dev_bleu"]) >= 10 and np.mean(metrics["dev_bleu"][-10:], axis=0)[1] > 20:
            break

    # Save final checkpoint
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path) 
    return metrics


if __name__ == "__main__":
    """
    Training script for BasicModel and AttentiveModel.
    
    Usage:
        python training.py basic          # Train BasicModel
        python training.py attentive      # Train AttentiveModel
    """

    def load_data():
        """Load and split data as in the notebook: 3000 dev samples."""
        data_dir = Path(__file__).parent / "data"
        data_inp_path = data_dir / "train.bpe.ru"
        data_out_path = data_dir / "train.bpe.en"

        if not data_inp_path.exists() or not data_out_path.exists():
            raise FileNotFoundError(
                f"Data files not found: {data_inp_path}, {data_out_path}\n"
                f"Please run: bash setup.sh data"
            )

        data_inp = np.array(open(data_inp_path).read().split("\n"))
        data_out = np.array(open(data_out_path).read().split("\n"))

        mask = (data_inp != "") & (data_out != "")
        data_inp, data_out = data_inp[mask], data_out[mask]

        train_inp, dev_inp, train_out, dev_out = train_test_split(
            data_inp, data_out, test_size=3000, random_state=42
        )
        return train_inp, dev_inp, train_out, dev_out

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_type = sys.argv[1].lower()

    train_inp, dev_inp, train_out, dev_out = load_data()
    inp_voc = Vocab.from_lines(train_inp)
    out_voc = Vocab.from_lines(train_out)

    if model_type == "basic":
        print("=" * 80)
        print("Training BasicModel")
        print("=" * 80)

        model = BasicModel(inp_voc, out_voc).to("cuda")
        metrics = train_model(
            model, train_inp, train_out, dev_inp, dev_out,
            inp_voc, out_voc, "cuda", 3000
        )

        final_bleu = metrics["dev_bleu"][-1][1]
        print(f"\n✓ BasicModel training complete!")
        print(f"  Final BLEU score: {final_bleu:.2f}")
        print(f"  Checkpoint saved to: outputs/basic_model_checkpoint.pt")

    elif model_type == "attentive":
        print("=" * 80)
        print("Training AttentiveModel")
        print("=" * 80)
        
        model = AttentiveModel("attentive", inp_voc, out_voc).to("cuda")
        metrics = train_attentive_model(
            model, train_inp, train_out, dev_inp, dev_out,
            inp_voc, out_voc, "cuda", 3000
        )
        
        final_bleu = metrics["dev_bleu"][-1][1]
        print(f"\n✓ AttentiveModel training complete!")
        print(f"  Final BLEU score: {final_bleu:.2f}")
        print(f"  Checkpoint saved to: outputs/attentive_model_checkpoint.pt")

    else:
        print(f"Unknown model type: {model_type}")
        print(__doc__)
        sys.exit(1)
