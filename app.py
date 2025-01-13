import streamlit as st
import pickle
import nltk
import gdown
import os
import helper

# تحميل الموارد إذا لم تكن موجودة
try:
    nltk.data.find('tokenizers/punkt')  # تحقق من وجود ملف punkt
except LookupError:
    nltk.download('punkt', quiet=True)  # إذا لم يكن موجودًا، قم بتحميله

try:
    nltk.data.find('tokenizers/punkt_tab')  # تحقق من وجود ملف punkt_tab
except LookupError:
    nltk.download('punkt_tab', quiet=True)  # إذا لم يكن موجودًا، قم بتحميله

# تحميل stopwords
nltk.download('stopwords', quiet=True)

# روابط نماذج Google Drive
MODEL_URLS = {
    "Logistic Regression": "https://drive.google.com/uc?id=1887AgoAPiU5QxcjAt6WJVC0sXfIbMXl1",
    "SVM": "https://drive.google.com/uc?id=17Y3eoOCUeEaN1CpIMCz3dFuslhafuJWK",
    "Decision Tree": "https://drive.google.com/uc?id=11WV6qGw2KTGDoEzOY_C9oBdBAQi--MH7"
}

# تحميل النموذج فقط مرة واحدة
@st.cache_resource
def load_model(model_name):
    """تحميل النموذج من Google Drive فقط مرة واحدة."""
    model_path = f"{model_name}.pkl"

    # تحميل النموذج إذا لم يكن موجودًا
    if not os.path.exists(model_path):
        gdown.download(MODEL_URLS[model_name], model_path, quiet=False)

    # تحميل النموذج من الملف
    return pickle.load(open(model_path, 'rb'))

# السماح للمستخدم باختيار النموذج
model_choice = st.selectbox("Select the model you want to use:", list(MODEL_URLS.keys()))

# تحميل النموذج المحدد (cached)
model = load_model(model_choice)

# إدخال المستخدم للنص
text = st.text_input("Enter your review:")

# معالجة النص المدخل
if text:
    processed_text = helper.text_preprocessing(text)

    # إجراء التنبؤ
    if st.button("Predict"):
        try:
            pred = model.predict(processed_text)[0]
            
            # عرض الرسالة المناسبة بناءً على التنبؤ
            if pred == 1:
                st.success("The review is Positive! 😊")
            else:
                st.error("The review is Negative! 😞")
        except Exception as e:
            st.error(f"Error processing the text: {e}")
