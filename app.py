# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import joblib
import spacy
import nltk; nltk.download('popular')
import en_core_web_sm
from Question import Question
import preprocessing as p
# 2. Create the app object
app = FastAPI()
vectorizer = joblib.load("tfidf_vectorizer.pkl", 'r')
multilabel_binarizer = joblib.load("multilabel_binarizer.pkl", 'r')
model = joblib.load("logisticreg_nlp_model.pkl", 'r')


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To autotag stackoverflow questions API': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data 
@app.post('/predict')
def predict(question:Question):
      
        # Clean the question sent
        
        nlp = en_core_web_sm.load(exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
        pos_list = ["NOUN","PROPN"]
        print('question entrée : ', question.question)
        cleaned_question = p.text_cleaner(question.question, nlp, pos_list, "english")

        
        
        # Apply saved trained TfidfVectorizer
        X_tfidf = vectorizer.transform([cleaned_question])
       
        
        # Perform prediction
        predict = model.predict(X_tfidf)
        predict_probas = model.predict_proba(X_tfidf)
        # Inverse multilabel binarizer
        tags_predict = multilabel_binarizer.inverse_transform(predict)
        
        
        df_predict = pd.DataFrame(columns=['Tags', 'Probas'])
        df_predict['Tags'] = multilabel_binarizer.classes_
        df_predict['Probas'] = predict_probas.reshape(-1)
        
        df_predict= df_predict[df_predict['Probas']>=0.20].sort_values('Probas', ascending=False)
            
        # Results
        results = {}
        results['Predicted_Tags'] = df_predict[df_predict['Probas']>=0.40].sort_values('Probas', ascending=False)
        #results['Predicted_Tags_Probabilities'] = df_predict.set_index('Tags')['Probas'].to_dict()
        
        return results, 200







# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload