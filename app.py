from flask import Flask,request,jsonify,render_template
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app=Flask(__name__)
ReviewLoadedModel, corpus2= joblib.load('reviewModel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    str_features=[str(x) for x in request.form.values()]
    review=re.sub("[^a-zA-Z]",' ',str_features[0])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()# lemmatization give root or base word call for calling,called
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]#
    review=" ".join(review)
    corpus=[]
    corpus.append(review)
    cv2=CountVectorizer(max_features=1500)
    test_x=cv2.fit_transform(corpus2+corpus).toarray()
    final_review=test_x[-1].reshape(1,-1)
    result=ReviewLoadedModel.predict(final_review)
    if result==1:
        output="Positive"
    else:
        output="Negative"
    #output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text='Review is {}'.format(output))


if __name__=="__main__":
    app.run()