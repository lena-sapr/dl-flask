from json import load
import os
from flask import Flask, redirect, render_template, request, flash, url_for
from werkzeug.utils import secure_filename
from model.cnnmodel import predict
from model.model_init import load_cnn_model, load_text_gen_model, load_text_model
from model.textmodel import generate_text, get_sentiment
import concurrent

UPLOAD_FOLDER = 'static/files/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)  #задаем создание приложения фласка
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = 2**16
app.secret_key = b'0000'

global sent
sent = {
    'positive' : 'позитивной',
    'negative' : 'негативной',
    'neutral'  : 'нейтральной'
}

# @app.before_first_request
# def init_models():
#     global cnn_model, text_model, tokenizer
#     cnn_model = load_cnn_model()
#     text_model, tokenizer = load_text_model()
#     thread = threading.Thread(target=init_models)
#     thread.start()

@app.before_first_request  #загружаем модели, даже если никто не заходил еще на сайт
def init_models():
    # задаем глобальные переменные, в которые запишем модели
    global cnn_model, text_model, tokenizer, text_generator_model, text_generator_tokenizer

    with concurrent.futures.ThreadPoolExecutor() as executor:  #модуль concurrent нужен для запуска нескольких потоков
        # создаем несколько потоков
        cnn_future = executor.submit(load_cnn_model)
        text_model_future = executor.submit(load_text_model)
        text_generator_model_future = executor.submit(load_text_gen_model)
        
        # забираем результат из каждого потока 
        cnn_model = cnn_future.result()
        text_model, tokenizer = text_model_future.result()
        text_generator_model, text_generator_tokenizer = text_generator_model_future.result()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')  #все страницы во фласке - это функции. Это функция для титульной (хоум) страницы
def index():
    return render_template('index.html')  #вызываеет рендеринг этого темплейта из папки темплейтс (папка должна называться именно templates)

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html', query=True)

@app.route('/task')
def task():
    return render_template('task.html')

@app.route('/text_generation')
def text_generation():
    return render_template('text_generation.html', query=True)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict(cnn_model, filepath)
        else:
            flash('Choose correct file!')
            return redirect(url_for('image'))
    return render_template('image.html', img=filepath, label=prediction)

@app.route('/analyzer', methods=['POST'])
def sent_analysis():
    if request.method == 'POST':
        text = request.form['sentiment']
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        result = get_sentiment(text_model, inputs)
        # print(result)

    return render_template('sentiment.html', result=sent[result], text=text)

@app.route('/text_generator', methods=['POST'])
def text_generator():
    if request.method == 'POST':
        
        prompt = request.form['prompt']
        length = request.form['length']
        num_texts = request.form['num_texts']
        temperature = request.form['temperature']
        use_radio = request.form['radio']
        print(use_radio)

        inputs = text_generator_tokenizer.encode(prompt, return_tensors='pt')
        out = generate_text(text_generator_model, inputs, length, num_texts, temperature, use_radio)

        res = []

        for out_ in out:
            
            out_ = text_generator_tokenizer.decode(out_)
            # print(out_)
            if '.' in out_:
                dot_ind = out_.rindex('.')
                # print(dot_ind)
                out_ = out_[:dot_ind]
                # print(out_)
            res.append(out_)
            # print(res)
           
    return render_template('text_generation.html', result=res)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 