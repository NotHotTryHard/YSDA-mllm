import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bokeh.plotting as pl
import bokeh.models as bm
from bokeh.io import output_notebook, show
from bokeh.layouts import row

from vocab import Vocab
from seq2seq import AttentiveModel


def draw_attention(inp_line, translation, probs, inp_voc, out_voc):
    """Visualize attention weights using matplotlib"""
    inp_token_indices = inp_voc.tokenize(inp_line)
    trans_token_indices = out_voc.tokenize(translation)
    probs = probs[:len(trans_token_indices), :len(inp_token_indices)]

    inp_token_strs = [list(inp_voc.mapper.keys())[idx] for idx in inp_token_indices]
    trans_token_strs = [list(out_voc.mapper.keys())[idx] for idx in trans_token_indices]

    fig, ax = plt.subplots(figsize=(max(8, len(inp_token_strs) * 0.5), max(6, len(trans_token_strs) * 0.4)))
    
    im = ax.imshow(probs[::-1], aspect='auto', cmap='Blues', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(inp_token_strs)))
    ax.set_yticks(np.arange(len(trans_token_strs)))
    ax.set_xticklabels(inp_token_strs, rotation=45, ha='right')
    ax.set_yticklabels(trans_token_strs[::-1])
    
    ax.set_xlabel('Source tokens', fontsize=12)
    ax.set_ylabel('Translation tokens', fontsize=12)
    ax.set_title('Attention Matrix', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Attention weight')
    plt.tight_layout()
    return fig


def extract_attention_probs(attentive_model, dev_inp):
    """Extract attention probabilities from the attentive model."""
    inp = dev_inp[::10]
    trans, states = attentive_model.translate_lines(inp)
    
    # Extract attention probabilities from states
    # your code here \/

    attn_probs_list = [state[-1] for state in states[1:]]
    attention_probs = torch.stack(attn_probs_list, dim=1).cpu().detach().numpy()
    # your code here /\
    
    return attention_probs, trans, inp


def load_model_and_vocabs(checkpoint_path="outputs/attentive_model_checkpoint.pt", data_dir="data"):
    checkpoint_path = Path(checkpoint_path)
    data_dir = Path(data_dir)
    
    data_inp_path = data_dir / "train.bpe.ru"
    data_out_path = data_dir / "train.bpe.en"
    
    data_inp = np.array(open(data_inp_path).read().split("\n"))
    data_out = np.array(open(data_out_path).read().split("\n"))
    
    from sklearn.model_selection import train_test_split
    train_inp, dev_inp, train_out, dev_out = train_test_split(
        data_inp, data_out, test_size=3000, random_state=42
    )
    
    inp_voc = Vocab.from_lines(train_inp)
    out_voc = Vocab.from_lines(train_out)
    
    model = AttentiveModel(
        'attentive', inp_voc, out_voc,
        emb_size=64, hid_size=128, attn_size=128
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to("cpu")
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"✓ Vocabularies created: {len(inp_voc)} input tokens, {len(out_voc)} output tokens")
    print(f"✓ Development set: {len(dev_inp)} sentences")
    return model, inp_voc, out_voc, dev_inp


def visualize_attention_maps(attentive_model, dev_inp, inp_voc, out_voc, num_examples=3, save_dir="outputs"):
    """Visualize attention maps for multiple translation examples and save as PNG."""
    attention_probs, translations, inputs = extract_attention_probs(attentive_model, dev_inp)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for i in range(min(num_examples, len(inputs))):
        fig = draw_attention(inputs[i], translations[i], attention_probs[i], inp_voc, out_voc)
        
        save_path = save_dir / f"attention_map_{i+1}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        saved_files.append(str(save_path))
        print(f"✓ Saved attention map {i+1} to {save_path}")
        print(f"  Input: {inputs[i][:80]}...")
        print(f"  Translation: {translations[i][:80]}...")
    
    return saved_files


if __name__ == "__main__":
    print("Loading model and vocabularies...")
    model, inp_voc, out_voc, dev_inp = load_model_and_vocabs(
        checkpoint_path="outputs/attentive_model_checkpoint.pt",
        data_dir="data"
    )
    print("\nGenerating attention visualizations...")
    saved_files = visualize_attention_maps(
        model, dev_inp, inp_voc, out_voc, 
        num_examples=3, 
        save_dir="outputs"
    )
    print(f"\n✓ Successfully generated {len(saved_files)} attention maps!")
    print(f"  Saved to: {saved_files}")
