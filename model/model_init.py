import torch
import torchvision
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel

def load_cnn_model():
    cnn_model = torchvision.models.resnet18(pretrained=True)
    cnn_model.eval()
    return cnn_model

def load_text_model():
    model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    text_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    text_model.eval()
    return text_model, tokenizer

def load_text_gen_model():
    model_checkpoint = 'sberbank-ai/rugpt3small_based_on_gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    text_gen_model = GPT2LMHeadModel.from_pretrained(model_checkpoint, output_attentions = False, output_hidden_states = False)
    text_gen_model.eval()
    return text_gen_model, tokenizer

