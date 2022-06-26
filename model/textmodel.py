import torch

def generate_text(model, inputs, length, num_texts, temperature, use_radio):
    # prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)

    if (use_radio == 'use_radio'):
        state_dict = torch.load('model/Radio_Adonezh.ptn', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    max_length = int(length)
    num_return_sequences = int(num_texts)
    temperature = float(temperature)
    print('temperature:', temperature)

    out = model.generate(
        input_ids=inputs,
        max_length= max_length,#50,
        num_beams=10,
        do_sample=True,
        temperature= 10.,
        top_k=50,
        top_p=0.6,
        no_repeat_ngram_size=2,
        num_return_sequences=num_return_sequences, #10,
        ).cpu().numpy()

    
    return out

def get_sentiment(model, inputs):
    proba = torch.sigmoid(model(**inputs).logits).cpu().detach().numpy()[0]
    return model.config.id2label[proba.argmax()]

