import os
import time
from flask import Flask, render_template, request

from predict import HandwrittenDigitsPredicter



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHECK_POINT_MODEL'] = './best.pth'


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        
        if 'file' in request.files:
            start_time = time.time()
            file = request.files['file']
            file_path_save = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path_save)

            image_path = file_path_save

            predicter = HandwrittenDigitsPredicter(image_path=image_path, check_point=app.config['CHECK_POINT_MODEL'])
            label_hat = predicter.predict()
            result = 'CAT' if label_hat == 0 else 'DOG'
            end_time = time.time()
            run_time = str(end_time - start_time)
            return render_template("layout-home.html", result=result, run_time=run_time)
    return render_template("layout-home.html")


if __name__ == '__main__':
    app.run(debug=True)