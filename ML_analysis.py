from ensurepip import bootstrap
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import pair_confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE

def main():
    st.title("Anti-Money Laundering App")
    st.sidebar.title("Anti-Money Laundering App")
    st.markdown("Identifying  Money Laundered Transaction")
    st.sidebar.markdown("Identifying  Money Laundered Transaction")

    ## fileuploader

    st.sidebar.markdown('---')
    st.sidebar.subheader('File Uploader')

   
            
    uploaded_file = st.sidebar.file_uploader('Choose a file')
    save_button = st.sidebar.button('save file')
    if save_button:
        if uploaded_file is not None:
            with open(os.path.join("C:\\Users\\Dell\\Desktop\Data Scientist\\Streamlit", uploaded_file.name),mode='wb') as f:
                f.write(uploaded_file.getbuffer())
                    
                st.sidebar.success('File uploaded successfully')
        else:
             st.sidebar.warning('Please select the file you want to upload')
            
        
   
        
    @st.cache(persist=True)
    def load_data():
        try:
            
            data = pd.read_csv(uploaded_file)
            labelencoder=LabelEncoder()
            for col in data.columns:
                data[col] = labelencoder.fit_transform(data[col])
            
            col_names1 = ['amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest','newbalanceDest']
            col_names2 = ['type','isFraud']
            features = data[col_names1]
            features2 = data[col_names2]
            scaler = StandardScaler().fit(features.values)
            features = scaler.transform(features.values)
            scaled_features = pd.DataFrame(features, columns = col_names1)
            new_data=pd.concat([features2,scaled_features], axis=1, ignore_index=False)
            return new_data 

        except:
            print("Please Upload file to start the program!!!")
    
            
    
    
    df = load_data()    
    class_names = ['IsFraud', 'NotFraud']
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Synthetic Mobile Transaction (Classification)")
        st.write(df)

    @st.cache(persist=True)
    def split(df):
        try:
            y_org = df.isFraud
            x_org = df.drop(columns=['isFraud'])
            sm = SMOTE(random_state = 42)
            x, y = sm.fit_resample(x_org,y_org)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
            return x_train, x_test, y_train, y_test
        except:
             print("Please Upload file to start the program!!!")


    try:
        x_train, x_test, y_train, y_test = split(df)
    except:
        print("Please Upload file to start the program!!!")

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

        

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision Trees", "Random Forest"))

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2) )
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2) )
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2) )
            plot_metrics(metrics)

    if classifier == 'Decision Trees':

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
           st.subheader("Decision Trees Results")
           model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
           model.fit(x_train, y_train)
           accuracy = model.score(x_test, y_test)
           y_pred = model.predict(x_test)
           st.write("Accuracy: ", accuracy.round(2))
           st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names ).round(2))
           st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2) )
           plot_metrics(metrics)
    

if __name__ == '__main__':
    main()
