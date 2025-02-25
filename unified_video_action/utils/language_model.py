from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, CLIPModel
import torch
import pdb


def get_text_model(task_name, language_emb_model):
    if language_emb_model == "clip":
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif language_emb_model == "flant5":
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            text_model = T5EncoderModel.from_pretrained("google/flan-t5-base")
    else:
        tokenizer = None
        text_model = None

    if "libero_10" in task_name:
        max_length = 30
    elif "umi" in task_name:
        max_length = 30
    else:
        max_length = 30

    return text_model, tokenizer, max_length


def extract_text_features(text_model, text_tokens, language_emb_model):
    with torch.no_grad():
        if language_emb_model == "clip":
            text_latents = text_model.get_text_features(**text_tokens)

        elif language_emb_model == "flant5":
            text_latents = text_model(text_tokens).last_hidden_state.detach()
            print("flant5 text_latents", text_latents.max())
            if torch.isnan(text_latents).any():
                print("NaNs detected in text_latents")
                pdb.set_trace()
        else:
            pdb.set_trace()

    return text_latents
