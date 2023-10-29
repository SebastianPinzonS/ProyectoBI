
import modelPro
import joblib
import numpy as np
import pandas as pd



    
modelw = modelPro.getModel()
tfidf_model =  joblib.load("modelLR.joblib")
ex ="Es importante destacar que, en un año de sequía, se espera que disminuyan todos los aportes, pero que aumente la demanda de agua de los cultivos y de la vegetación, lo que provocará una reducción del caudal de los ríos que están conectados hidrológicamente con el acuífero, así como una menor disponibilidad de agua para otros ecosistemas dependientes de las aguas subterráneas. Estas externalidades espaciales pueden incluir el agotamiento de los cursos de agua y el descenso del nivel freático local. Por ejemplo, el bombeo continuo del pozo B provocará el agotamiento de los cursos de agua y la reducción de su caudal."

predict = modelPro.predict(ex,modelw,tfidf_model)
print(predict[0])
        