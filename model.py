# Importing the libraries
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 


#load data
quart = pd.read_excel('Data/raman.xlsx', sheet_name='quartzo') #quartz 45 ccw 
gold = pd.read_excel('Data/raman.xlsx', sheet_name='gold') #gold 785 nm
kh2po4 = pd.read_excel('Data/raman.xlsx', sheet_name='KH2PO4')#KH2PO4 (Potassium Dihydrogen Phosphate R120165) 780 nm
iron = pd.read_excel('Data/raman.xlsx', sheet_name='iron')#Iron 785 nm
diamond = pd.read_excel('Data/raman.xlsx', sheet_name='diamante')#diamante 45 ccw

mine = pd.concat([gold, quart, kh2po4, iron, diamond])
mine = pd.DataFrame(mine)

mineral_1 = ['Gold']
list_gold = mineral_1*1971

mineral_2 = ['Quartz']
list_quartz = mineral_2*1151

mineral_3 = ['Potassium Dihydrogen Phosphate']
list_potassium = mineral_3*2451

mineral_4 = ['Iron']
list_Iron = mineral_4*2237	

mineral_5 = ['Diamond']
list_d = mineral_5*1289


list_gold = pd.DataFrame(list_gold)
list_quartz = pd.DataFrame(list_quartz)
list_potassium = pd.DataFrame(list_potassium)
list_Iron = pd.DataFrame(list_Iron)
list_d = pd.DataFrame(list_d)

minerals = pd.concat([list_gold,list_quartz,list_potassium,list_Iron,list_d])
minerals = pd.DataFrame(minerals)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
mine['Minerals'] = minerals

mine['Minerals'] = mine['Minerals'].replace('Gold', 0)
mine['Minerals'] = mine['Minerals'].replace('Quartz', 1)
mine['Minerals'] = mine['Minerals'].replace('Potassium Dihydrogen Phosphate', 2)
mine['Minerals'] = mine['Minerals'].replace('Iron', 3)
mine['Minerals'] = mine['Minerals'].replace('Diamond', 4)

y = mine['Minerals']
x = mine.drop('Minerals', axis = 1) 

from sklearn.model_selection import train_test_split
x_trein, x_test, y_trein, y_test = train_test_split(x, y, test_size = 0.3) 

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x_trein, y_trein)
result = model.score(x_test, y_test)
print('Acuracy: {}'.format(result))
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''