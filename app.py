import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv() 

genai.configure(api_key=os.getenv(st.secrets["GOOGLE_API_KEY"]))
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input_prompt, user_input):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([input_prompt, user_input])
    return response.text



#Prompt Template

input_prompt="""
You are expert in field of breast cancer, you solve queries of users
"""

## streamlit app
st.set_page_config(layout="wide", page_title="Breast Cancer Diagnosis", page_icon="")
st.title("BREAST CANCER DIAGNOSIS")


col1, col2 = st.columns(2)
with col1:
    text1='''
    In India, breast cancer is a significant health concern, with approximately 1 in 28 women at risk of developing breast cancer during their lifetime. The incidence of breast cancer in India is on the rise, with an estimated 162,468 new cases diagnosed in 2020. 
    '''
    st.write(text1)
    text2 = 'Early detection and access to quality healthcare services are crucial in improving outcomes for breast cancer patients in India.'
    st.write(text2)
    text3="It’s important to understand that most breast lumps are benign and not cancer malignant. Non-cancer breast tumors are abnormal growths, but they do not spread outside of the breast. They are not life threatening, but some types of benign breast lumps can increase a woman's risk of getting breast cancer."
    st.write(text3)

    st.write('<h5>Dataset</h5>',unsafe_allow_html=True)
    st.write('We are using Breast Cancer Wisconsin dataset for analysis and diagnosis')
    st.markdown("[Link Text](https://retr0sushi04.notion.site/3-Breast-Cancer-Wisconsin-Diagnostic-3f0f8e047adc40248228376e6005c9fe?pvs=4)")
    st.write('This dataset contains features computed from images of fine needle aspirates of breast masses, describing characteristics of cell nuclei. ')

    st.write('<h5>Machine Learning Models</h5>',unsafe_allow_html=True)

    st.write('In our study on breast cancer diagnosis, various machine learning models were trained and evaluated for their performance.')

    st.write('Among the models assessed, the AdaBoostClassifier stood out with the highest accuracy of 98.2%, establishing it as the top-performing model for breast cancer diagnosis in our research. ')

    models = [
    "Logistic Regression",
    "Support Vector Classifier (SVC)",
    "K-Nearest Neighbors (KNN)",
    "Random Forest",
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "AdaBoost"
]

    selected_model = st.selectbox('Choose a model:', models)
    lr_code = '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    clf1 = LogisticRegression()
    clf1.fit(X_train, y_train)
    '''

    svc_code='''
    from sklearn.svm import SVC
    clf2 = SVC(kernel='linear')  

    # Training the model
    clf2.fit(X_train, y_train)
    '''

    knn_code='''
    from sklearn.neighbors import KNeighborsClassifier

    clf3 = KNeighborsClassifier(n_neighbors=5)
    clf3.fit(X_train, y_train)
    '''

    rf_code='''
    from sklearn.ensemble import RandomForestClassifier

    clf4 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf4.fit(X_train, y_train)
    '''

    xgb_code='''
    import xgboost as xgb

    clf5 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    clf5.fit(X_train, y_train)
    '''

    cb_code='''
    from catboost import CatBoostClassifier

    clf6 = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='Logloss', random_state=42)
    clf6.fit(X_train, y_train, cat_features=None)
    '''

    lb_code='''
    import lightgbm as lgb

    clf7 = lgb.LGBMClassifier()
    clf7.fit(X_train, y_train)
    '''

    adb_code='''
    from sklearn.ensemble import AdaBoostClassifier

    clf8 = AdaBoostClassifier()
    clf8.fit(X_train, y_train)
    '''


    if selected_model == models[0]:
        st.code(lr_code, language='python')
        st.write('Accuracy: 0.964')
    elif selected_model == models[1]:
        st.code(svc_code, language='python')
        st.write('Accuracy: 0.964')
    elif selected_model == models[2]:
        st.code(knn_code, language='python')
        st.write('Accuracy: 0.956')
    elif selected_model == models[3]:
        st.code(rf_code, language='python')
        st.write('Accuracy: 0.956')
    elif selected_model == models[4]:
        st.code(xgb_code, language='python')
        st.write('Accuracy: 0.956')
    elif selected_model == models[5]:
        st.code(cb_code, language='python')
        st.write('Accuracy: 0.965')
    elif selected_model == models[6]:
        st.code(lb_code, language='python')
        st.write('Accuracy: 0.974')
    elif selected_model == models[7]:
        st.code(adb_code, language='python')
        st.write('Accuracy: 0.982')


    st.image('model_acc.png', caption='')

    
    st.write('<h5>Tableau</h5>',unsafe_allow_html=True)

    st.write('<h6>Overview:</h6>',unsafe_allow_html=True)
    st.write('The goal of our analysis is to explore the characteristics of breast cancer tumors as represented in the dataset and uncover insights that could aid in diagnosis and treatment decisions.')

    st.write('<h6>Key Findings:</h6>',unsafe_allow_html=True)
    st.write("The dataset contains a total of 569 samples, with 212 (37.3%) being malignant and 357 (62.7%) being benign.The mean radius of malignant tumors is significantly larger than that of benign tumors, indicating that tumor size may be a key factor in classification.There are strong correlations between several features, such as radius and area, which is expected as these features are likely to be proportional.Features such as texture, smoothness, and concavity show noticeable differences between malignant and benign tumors, suggesting they could be important in distinguishing between the two.")
    
    st.write('<h6>Visualization:</h6>',unsafe_allow_html=True)
    st.video('Tableau - Datathon 2024-03-16 11-47-36.mp4')


with col2:
    st.info("Chat Below")
    user_question = st.chat_input("Ask your Query here...")
    if user_question:
        st.info(user_question)
        response = get_gemini_response(input_prompt,user_question)
        st.info(response)