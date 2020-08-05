from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
model = pickle.load(open('fakemodel.pickle', 'rb'))
@app.route('/')
def home():
    return render_template('fakenews.html')
@app.route('/predictbtn',methods=['POST'])
def predictbtn():
    text=request.form['textarea1']
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.75)
    def find_label(text):
        vec_newtest=tfidf_vectorizer.transform([text])
        y=model.predict(vec_newtest)
        return y[0]
    val1=find_label(text)
    return render_template('fakenews.html', prediction_text=val1)