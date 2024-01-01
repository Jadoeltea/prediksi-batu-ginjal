import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from function import train_model  

# Fungsi untuk menyiapkan data berdasarkan usia
def prepare_data(data):
    data['Age Group'] = pd.cut(data['age'], bins=range(0, 100, 10), labels=[f'{i}-{i+9}' for i in range(0, 90, 10)])
    age_counts = data['Age Group'].value_counts().sort_index()
    return age_counts

def app(data, x, y):
    st.title("Visualisasi Prediksi Batu Ginjal")

    age_data = prepare_data(data)

    if st.checkbox("Grafik Penderita Batu Ginjal Berdasarkan Usia"):
        plt.figure(figsize=(10, 6))
        age_data.plot(kind='bar', color='skyblue')
        plt.xlabel('Usia')
        plt.ylabel('Jumlah Penderita Batu Ginjal')
        plt.title('Jumlah Penderita Batu Ginjal Berdasarkan Usia')
        st.pyplot()

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x, y)
        predictions = model.predict(x)
        cm = confusion_matrix(y, predictions)
        plt.figure(figsize=(10, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=4, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['nockd', 'ckd']
        )
        st.graphviz_chart(dot_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
