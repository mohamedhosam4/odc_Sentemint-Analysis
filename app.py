import streamlit as st
import pickle
import nltk
import gdown
import os
import helper
import numpy as np

# تحميل الموارد اللازمة من NLTK
try:  
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

nltk.download('stopwords', quiet=True)

# روابط النماذج على Google Drive (قم بتعديل الروابط إلى الروابط الخاصة بك)
MODEL_URLS = {
    "Logistic Regression": "https://drive.google.com/uc?id=1887AgoAPiU5QxcjAt6WJVC0sXfIbMXl1",
    "SVM": "https://drive.google.com/uc?id=17Y3eoOCUeEaN1CpIMCz3dFuslhafuJWK",
    "Decision Tree": "https://drive.google.com/uc?id=11WV6qGw2KTGDoEzOY_C9oBdBAQi--MH7"
}

@st.cache_resource
def load_model(model_name):
    """تحميل النموذج من Google Drive مرة واحدة فقط."""
    model_path = f"{model_name}.pkl"

    # تحميل النموذج إذا لم يكن موجوداً
    if not os.path.exists(model_path):
        gdown.download(MODEL_URLS[model_name], model_path, quiet=False)

    # تحميل النموذج من الملف
    return pickle.load(open(model_path, 'rb'))

# السماح للمستخدم باختيار النموذج الذي يود استخدامه
model_choice = st.selectbox("Select the model you want to use:", list(MODEL_URLS.keys()))

# تحميل النموذج المختار (مخبأ)
model = load_model(model_choice)

# إدخال النص من المستخدم
text = st.text_input("Enter your review:")

# معالجة النص المدخل
if text:
    try:
        # معالجة النص باستخدام الدالة في helper.py
        processed_text = helper.text_preprocessing(text)

        # تحويل processed_text إلى الشكل المناسب (يجب أن يكون مصفوفة من المتغيرات)
        # على سبيل المثال، إذا كنت تستخدم CountVectorizer أو TfidfVectorizer يجب أن يكون لديك هذه العملية
        # هنا نفترض أنه يجب عليك تحويل النص إلى مصفوفة بعد المعالجة
        processed_text = np.array([processed_text])

        # التنبؤ باستخدام النموذج
        if st.button("Predict"):
            pred = model.predict(processed_text)[0]

            # عرض الرسالة المناسبة بناءً على التنبؤ
            if pred == 1:
                st.success("The review is Positive! 😊")
            else:
                st.error("The review is Negative! 😞")

    except Exception as e:
        st.error(f"Error processing text: {e}")
