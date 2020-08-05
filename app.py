from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('fakemodel.pickle', 'rb'))
vect = pickle.load(open('vectorizer1.pickle', 'rb'))
@app.route('/')
def home():
    return render_template('fakenews.html')

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['textarea1']
    def find_label(text):
        vec_newtest=vect.transform([text])
        y=model.predict(vec_newtest)
        return y[0]
    val1=find_label(text)
    return render_template('fakenews.html', prediction_text=val1)


if __name__ == "__main__":
    app.run(debug=True)