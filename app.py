
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from ultralytics import YOLO
from PIL import Image
from datetime import datetime , timedelta
from real import generate_forecast
import warnings
import base64
import tempfile



import sqlite3
import qrcode
from cryptography.fernet import Fernet
import json
import uuid
import io
from fpdf import FPDF
import os
import logging

warnings.filterwarnings('ignore')



# إعداد التسجيل لتتبع الأخطاء
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# إعداد التشفير لرموز QR
key_file = "fernet_key.key"
if os.path.exists(key_file):
    with open(key_file, "rb") as f:
        key = f.read()
else:
    key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(key)
cipher = Fernet(key)

# إعداد قاعدة البيانات
def init_db():
    try:
        conn = sqlite3.connect("crowd_management.db")
        cursor = conn.cursor()

        # جدول المعتمرين
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pilgrims (
                id TEXT PRIMARY KEY,
                entry_time TEXT,
                floor TEXT,
                exit_time TEXT
            )
        ''')

        # جدول الأدوار
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS floors (
                floor TEXT PRIMARY KEY,
                current_count INTEGER,
                max_capacity INTEGER,
                status TEXT,
                last_updated TEXT
            )
        ''')

        # جدول سجل الأنشطة
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pilgrim_id TEXT,
                action TEXT,
                floor TEXT,
                timestamp TEXT
            )
        ''')

        cursor.execute("INSERT INTO activity_log (pilgrim_id, action, floor, timestamp) VALUES (?, ?, ?, ?)",
               ("test_pilgrim_1", "Entry", "Ground", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        cursor.execute("INSERT INTO activity_log (pilgrim_id, action, floor, timestamp) VALUES (?, ?, ?, ?)",
               ("test_pilgrim_2", "Exit", "Ground", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # إضافة بيانات افتراضية للأدوار
        cursor.execute('''
            INSERT OR IGNORE INTO floors (floor, current_count, max_capacity, status, last_updated)
            VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
        ''', (
            'Ground', 0, 50000, 'Available', datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'First', 0, 30000, 'Available', datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Second', 0, 20000, 'Available', datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        logger.info("Database initialized successfully with tables: pilgrims, floors, activity_log")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        st.error(f"خطأ في إعداد قاعدة البيانات: {e}")
    finally:
        conn.close()

# إنشاء مجلد لتخزين رموز QR
if not os.path.exists("qr_codes"):
    os.makedirs("qr_codes")

# إعداد قاعدة البيانات عند بدء التطبيق
init_db()


# Page configuration
st.set_page_config(page_title="Makkah Crowd Management Dashboard", layout="wide")

# Initialize session state for language and forecast data
if 'language' not in st.session_state:
    st.session_state.language = 'English'  # Changed from 'Arabic' to 'English'
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None

# Language-specific content
content = {
    'Arabic': {
        'team': {
            'title': " إدارة الحشود في مكة المكرمة",
            'overview': "مشروع إدارة الحشود في مكة المكرمة هو مبادرة تقنية رائدة تعتمد على الذكاء الاصطناعي لتحسين إدارة الحشود خلال الحج والعمرة. من خلال تحليل البيانات ونماذج التعلم الآلي، نتوقع كثافة الحشود في الطواف، السعي، ومناطق أخرى، مما يمكّن السلطات من تقليل المخاطر، تحسين الحركة، وضمان تجربة أكثر أمانًا وتنظيمًا لملايين الحجاج.",
            'target_audience': [
    {
        "target": "إدارة الحرم المكي",
        "img": "https://img.icons8.com/?size=512w&id=41170&format=png&color=40C057",
        "benefit": "تحسين التخطيط التشغيلي من خلال توقعات دقيقة لكثافة الحشود، إدارة تدفق الحجاج بكفاءة، وتعزيز تجربة الحجاج بتقليل الازدحام وضمان بيئة آمنة."
    },
    {
        "target": "السلطات الأمنية",
        "img": "https://img.icons8.com/?size=512w&id=41170&format=png&color=40C057",
        "benefit": "الكشف المبكر عن المخاطر، تحسين توزيع القوى الأمنية بناءً على التوقعات، وتقليل الحوادث الأمنية مثل التدافع."
    },
    {
        "target": "فرق الطوارئ",
        "img": "https://img.icons8.com/?size=512w&id=41170&format=png&color=40C057",
        "benefit": "الاستجابة السريعة للحوادث باستخدام خرائط تفاعلية، توقع المناطق الحرجة للاستعداد المسبق، وتحسين إدارة الموارد الطبية."
    }
],

            'expected_results': [
                {'result' : "تقليل الحوادث  " , 'number': 20 },
                {'result' : "تحسين كفاءة التنقل داخل الحرم  " , 'number': 30 },
                {'result' : "تقليل وقت الاستجابة للطوارئ " , 'number': 10 },

            ],


            'map_title': "مناطق إدارة الحشود في مكة",
            'team_title': "تعرف على فريقنا",
            'team': [
                {"name": "العنود نايف", "emoji": ":purple_heart:","more":"Team Leader" ,  "linkedin": "https://www.linkedin.com/in/alanoud-razin-98aa6b2a7/"},
                {"name": "معالي الخالدي", "emoji": ":blue_heart:","more":"Team Member" , "linkedin": "https://www.linkedin.com/in/maali-alkhaldi-991967215/"},
                {"name": "سارة العتيبي", "emoji": ":green_heart:", "more":"Team Member" ,"linkedin": "https://www.linkedin.com/in/sarah-alotaibi-6576921a7?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app"},
                {"name":"اثير الكناني", "emoji": ":white_heart:", "more":"Team Member" ,"linkedin": "https://www.linkedin.com/in/jana-almujally-031a5223b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"},
                {"name": " أبرار الدوسري", "emoji": ":yellow_heart:","more":"Team Member" , "linkedin": "http://linkedin.com/in/abrar-aldosari-592199215"}
            ],
            'map_locations': ["الكعبة", "الصفا", "المروة", "الطواف"]
        },
        'video': {
            'title': "معرض فيديوهات إدارة الحشود",
            'description': "استكشف مقاطع الفيديو التي تعرض ديناميكيات الحشود واستراتيجيات الإدارة في مكة المكرمة.",
            'captions': [
                "تحليل كثافة الحشود في الطواف",
                "تحليل كثافة الحشود في الطواف",
                "محيط الكعبة",
                "تدفق الحشود أثناء السعي"
            ],
            'error': "ملف الفيديو {file} غير موجود. تأكد من وجوده في مجلد 'videos'."
        },
        'prediction': {
            'title': "توقع مستوى الحشود",
            'description': "توقع مستويات الحشود في مكة المكرمة للأيام القادمة لدعم الإدارة الفعالة.",
            'start_date': "اختر تاريخ بدء التوقع",
            'forecast_days': "اختر مدة التوقع (أيام)",
            'predict_button': "توقع",
            'forecast_title': "{days}-توقع الحشود ليوم",
            'prediction_time': "بناءً على التوقع المطلوب في: {time}",
            'select_day': "اختر يومًا لعرض توقع الحشود على الخريطة",
            'map_title': "توزيع الحشود المتوقعة في مكة",
            'download_button': "تحميل التوقعات",
            'error': "خطأ في جلب التوقع: {error}",
            'info': "اختر تاريخ البدء ومدة التوقع، ثم اضغط 'توقع' لإنشاء توقع الحشود. اختر يومًا لعرض التوقع على الخريطة."
        },
        'about' : {
    'title': "عن المشروع",
    'overview': "نبذة مختصرة",
    'overview_text': "مشروع إدارة الحشود في مكة المكرمة هو مبادرة تقنية تم تطويرها ضمن معسكر علم البيانات، ويهدف إلى تعزيز سلامة وكفاءة إدارة الحشود خلال مواسم الحج والعمرة باستخدام تقنيات الذكاء الاصطناعي وعلم البيانات.",
    'vision': "رؤيتنا",
    'vision_text': "توفير بيئة آمنة ومنظمة للحجاج والزوار في مكة المكرمة من خلال الاستفادة من أحدث تقنيات الذكاء الاصطناعي لتحسين إدارة الحشود وضمان تجربة سلسة للجميع.",
    'mission': "مهمتنا",
    'mission_text': "تطوير نظام ذكي يعتمد على تحليل البيانات في الوقت الفعلي لتوقع كثافة الحشود وتحديد المناطق الحرجة، مما يساهم في تقليل المخاطر وتحسين استجابة الطوارئ.",
    'goals': "أهدافنا",
    'goals_list': [
        "تقليل الحوادث الناتجة عن الازدحام بنسبة 20%.",
        "تحسين كفاءة التنقل داخل الحرم المكي بنسبة 30%.",
        "تقليل أوقات الاستجابة للطوارئ من خلال توقعات دقيقة.",
        "توفير بيانات دقيقة للسلطات لاتخاذ قرارات فعالة."
    ],
    'technical_approach': "الجانب التقني",
    'technical_approach_text': "يعتمد المشروع على منهجية علم البيانات الشاملة التي تشمل الخطوات التالية:",
    'technical_approach_list': [
        "جمع البيانات: تم جمع بيانات الحشود التاريخية من مصادر متعددة، بما في ذلك سجلات الحرم المكي وبيانات الكاميرات الأمنية.",
        "المعالجة المسبقة للبيانات: قمنا بتنظيف البيانات ومعالجتها للتعامل مع القيم المفقودة والضوضاء، وتحويل بيانات الفيديو إلى تنسيق قابل للتحليل باستخدام تقنيات معالجة الصور.",
        "بناء النماذج: استخدمنا نماذج التعلم الآلي مثل السلاسل الزمنية (Time Series) لتوقع كثافة الحشود، ونماذج التعلم العميق (مثل الشبكات العصبية الالتفافية CNN) لتحليل بث الفيديو المباشر واكتشاف الكثافة في الوقت الفعلي.",
        "التحليل والتصور: طوّرنا لوحة تحكم تفاعلية باستخدام Streamlit وPlotly لعرض التوقعات والتحليلات بشكل مرئي لدعم اتخاذ القرار."
    ],
    'technologies': "التقنيات المستخدمة",
    'technologies_list': [
        "Python: لغة البرمجة الأساسية لتطوير النماذج وتحليل البيانات.",
        "مكتبات علم البيانات: Pandas وNumPy لمعالجة البيانات، وScikit-learn لبناء نماذج التعلم الآلي.",
        "التعلم العميق: TensorFlow وKeras لتطوير نماذج تحليل الفيديو.",
        "تصور البيانات: Plotly وStreamlit لإنشاء لوحات تحكم تفاعلية.",
        "معالجة الصور: OpenCV لتحليل بيانات الفيديو."
    ],
    'value': "القيمة المضافة",
    'value_text': "يجمع مشروعنا بين علم البيانات والذكاء الاصطناعي لتقديم حلول مبتكرة تلبي احتياجات إدارة الحشود في مكة، مما يعزز السلامة العامة ويحسن تجربة الحجاج والزوار.",
    'chatbot': "بوت دليل الحرم",
    'chatbot_text' : "فتح بوت دليل الحرم",
},
            'realtime': {
            'title': "الكشف عن الحشود في الوقت الحقيقي",
            'description': "مراقبة الحشود في الوقت الحقيقي باستخدام كاميرا الويب ونموذج YOLOv8 للكشف عن الأشخاص وتقدير نسبة التغطية.",
            'start_button': "بدء الكشف",
            'stop_button': "إيقاف الكشف",
            'error': "فشل في تهيئة كاميرا الويب. تأكد من اتصال الكاميرا أو اختر مؤشر كاميرا مختلف.",
            'warning': "لا يمكن استقبال الإطار من الكاميرا. يتم الإنهاء..."
        },

        'nav': {
            'language_label': "اللغة",
            'pages': ['تعريف بالفريق', 'توقع الحشود', 'مركز المراقبة', 'تخصيص الموارد', 'أسئلة شائعة', 'عن المشروع', 'إدارة الأدوار', 'التقارير']

        }
    },
    'English': {
        'team': {
            'title': "SafeCrowd  ",
            'overview': "The Makkah Crowd Management Project is an innovative AI-driven initiative to enhance crowd management during Hajj and Umrah. Using advanced data analytics and machine learning, we predict crowd density in Tawaf, Saei, and other areas, enabling authorities to reduce risks, optimize movement, and ensure a safer, more efficient experience for millions of pilgrims.",
            'target_audience': [

    {
        "target": "Makkah Haram Administration",
        "img": "https://img.icons8.com/?size=512w&id=41170&format=png&color=40C057",
        "benefit": "Enhanced operational planning with accurate crowd predictions, efficient pilgrim flow management, and an improved pilgrim experience by reducing congestion and ensuring safety."
    },
    {
        "target": "Security Authorities",
        "img": "https://img.icons8.com/?size=512w&id=41170&format=png&color=40C057",
        "benefit": "Early risk detection, optimized deployment of security forces based on predictions, and reduced security incidents like stampedes."
    },
    {
        "target": "Emergency Response Teams",
        "img": "https://img.icons8.com/?size=512w&id=41170&format=png&color=40C057",
        "benefit": "Faster incident response with interactive maps, anticipation of critical areas for proactive preparation, and improved medical resource management."
    }
],


            'expected_results': [
                {'result' : "Reduce incidents  " , 'number': 20 },
                {'result' : "Improve movement efficiency within the Haram  " , 'number': 30 },
                {'result' : "Reduce emergency response time  " , 'number': 10 },

            ],
            'map_title': "Makkah Crowd Management Zones",
            'team_title': "Meet Our Team",
            'team': [
                {"name": "Alanoud Naif", "emoji": ":purple_heart:", "more":"Team Leader","linkedin": "https://www.linkedin.com/in/alanoud-razin-98aa6b2a7/"},
                {"name": "Maali Alkhaldi", "emoji": ":blue_heart:", "more":"Team Member","linkedin": "https://www.linkedin.com/in/maali-alkhaldi-991967215/"},
                {"name": "Sarah Alotaibi", "emoji": ":green_heart:", "more":"Team Member","linkedin": "https://www.linkedin.com/in/sarah-alotaibi-6576921a7?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app"},
                {"name": "ِAtheer Alkenani", "emoji": ":white_heart:", "more":"Team Member","linkedin": "https://www.linkedin.com/in/jana-almujally-031a5223b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"},
                {"name": "Abrar Aldosari", "emoji": ":yellow_heart:", "more":"Team Member","linkedin": "http://linkedin.com/in/abrar-aldosari-592199215"}
            ],
            'map_locations': ["Kaaba", "Safa", "Marwa", "Tawaf"]
        },
        'video': {
            'title': "Crowd Management Video Gallery",
            'description': "Explore videos showcasing crowd dynamics and management strategies in Makkah.",
            'captions': [
                "Tawaf Crowd Density Analysis",
                "Tawaf Crowd Density Analysis 2",
                "The perimeter of the Kaaba",
                "Crowd flow during Sa'i"

            ],
            'error': "Video file {file} not found. Ensure it exists in the 'videos' folder."
        },
        'prediction': {
            'title': "Crowd Level Prediction",
            'description': "Predict crowd levels in Makkah for the upcoming days to aid in effective management.",
            'start_date': "Select Start Date for Prediction",
            'forecast_days': "Select Forecast Period (Days)",
            'predict_button': "Predict",
            'forecast_title': "{days}-Day Crowd Forecast",
            'prediction_time': "Based on prediction triggered at: {time}",
            'select_day': "Select a Day to View Crowd Forecast on Map",
            'map_title': "Predicted Crowd Distribution in Makkah",
            'download_button': "Download Forecast",
            'error': "Error fetching prediction: {error}",
            'info': "Select a start date and forecast period, then click 'Predict' to generate a crowd forecast. Choose a day to view the forecast on the map."
        },
        'about' : {
    'title': "About the Project",
    'overview': "Project Overview",
    'overview_text': "The Makkah Crowd Management Project is a tech initiative developed as part of a Data Science Bootcamp, aimed at enhancing safety and efficiency in crowd management during Hajj and Umrah seasons using AI and data science techniques.",
    'vision': "Our Vision",
    'vision_text': "To provide a safe and organized environment for pilgrims and visitors in Makkah by utilizing cutting-edge AI technologies to improve crowd management and ensure a seamless experience for all.",
    'mission': "Our Mission",
    'mission_text': "To develop an intelligent system that relies on real-time data analysis to predict crowd density and identify critical areas, contributing to risk reduction and improved emergency response.",
    'goals': "Our Goals",
    'goals_list': [
        "Reduce crowd-related incidents by 20%.",
        "Enhance movement efficiency within the Haram by 30%.",
        "Minimize emergency response times through accurate predictions.",
        "Provide authorities with precise data for effective decision-making."
    ],
    'technical_approach': "Technical Approach",
    'technical_approach_text': "The project follows a comprehensive data science methodology, including the following steps:",
    'technical_approach_list': [
        "Data Collection: Historical crowd data was gathered from multiple sources, including Haram records and security camera feeds.",
        "Data Preprocessing: We cleaned and processed the data to handle missing values and noise, and transformed video data into an analyzable format using image processing techniques.",
        "Model Development: We used machine learning models such as Time Series forecasting to predict crowd density, and deep learning models (e.g., Convolutional Neural Networks - CNNs) for real-time video analysis and density detection.",
        "Analysis and Visualization: We developed an interactive dashboard using Streamlit and Plotly to visualize predictions and insights, aiding decision-making."
    ],
    'technologies': "Technologies Used",
    'technologies_list': [
        "Python: The primary programming language for model development and data analysis.",
        "Data Science Libraries: Pandas and NumPy for data processing, Scikit-learn for machine learning models.",
        "Deep Learning: TensorFlow and Keras for developing video analysis models.",
        "Data Visualization: Plotly and Streamlit for creating interactive dashboards.",
        "Image Processing: OpenCV for video data analysis."
    ],
    'value': "Added Value",
    'value_text': "Our project integrates data science and AI to deliver innovative solutions for crowd management in Makkah, enhancing public safety and improving the experience of pilgrims and visitors.",
    'chatbot': "Haram Guide Bot",
    'chatbot_text' : "Open Haram Guide Bot",
},

      'realtime': {
            'title': "Real-Time Crowd Detection",
            'description': "Monitor crowds in real-time using a webcam and YOLOv8 model to detect people and estimate coverage percentage.",
            'start_button': "Start Detection",
            'stop_button': "Stop Detection",
            'error': "Failed to initialize webcam. Ensure the webcam is connected or select a different webcam index.",
            'warning': "Can't receive frame from webcam. Exiting..."
        },

        'nav': {
            'language_label': "Language",
            'pages': ['Team Introduction', 'Crowd Prediction', 'Control Center', 'Resource Allocation', 'FAQ', 'About', 'Floor Management', 'Reports']
        }
    }
}

# Global CSS with st.tabs for navigation
global_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto Sans Arabic:wght@400;700&display=swap');
.stApp {
    background-color: #000000;  /* Solid black background */
    color: #ffffff ;  /* White text */
    font-family: 'Noto Sans Arabic', sans-serif;
    padding-top: 60px;  /* Space for fixed nav bar */
}

p, div, h1, h2, h3, h4, h5, h6 {
    text-align: center;
    color: #ffffff;  /* Green for headings */
}
.arabic-text {
    font-family: 'Noto Sans Arabic', sans-serif;
    text-align:center;
    color: #00ff00;


}
.nav-bar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: rgba(0, 0, 0, 0.8);  /* Semi-transparent black */
    padding: 0.5rem 1rem;
    z-index: 1000;
    display: flex;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.nav-content {
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: space-between;
}
/* Style the Streamlit tabs */
div.stTabs [data-baseweb="tab-list"] {
    display: flex;
    gap: 1.5rem;
    align-items: center;
}
div.stTabs [data-baseweb="tab"] {
    color: #ffffff;
    background-color: transparent;
    font-family: 'Noto Sans Arabic', sans-serif;
    font-size: 1rem;
    padding: 0.5rem 1rem;
    border: none;
    transition: color 0.3s ease;
}
div.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #00ff00;  /* Green for active tab */
    font-weight: bold;
    border-bottom: 2px solid #00ff00;
}
div.stTabs [data-baseweb="tab"]:hover {
    color: #00ff00;  /* Green on hover */
}
/* Style the language selector */
.language-selector {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.language-selector img {
    width: 24px;
}
.language-selector select {
    background-color: transparent;
    color: #ffffff;
    border: 1px solid #00ff00;
    border-radius: 15px;
    padding: 0.3rem 0.5rem;
    font-family: 'Noto Sans Arabic', sans-serif;
    cursor: pointer;
}
.hero-section {
    background-image: url('https://wallpaperaccess.com/full/3109193.jpg');
    background-size: cover;
    background-position: center;
    padding: 13rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    position: relative;
    z-index: 1;

    height: 600px; /* Adjust height as needed */
    display: flex;
    align-items: center;
    justify-content: center;
    color: red; /* Text color for contrast */
    text-align: center;
    overflow: hidden;
    margin-bottom: 10rem;

}
.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3);  /* Slight overlay for text readability */
    z-index: -1;
    border-radius: 10px;
    margin-bottom: 10rem;

}


.about-card {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 255, 0, 0.3);
    transition: transform 0.2s, box-shadow 0.2s;

    width: 220px !important;
    height: 120px !important;
    margin-bottom: 1.5rem;
}

.video-card {

    background-color: rgba(255, 255, 255, 0.1);
    padding: 0.5rem;
    border-radius: 5px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s, box-shadow 0.2s;

}


.team-card {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 1.5rem;
    width: 120 !important;
    height: 270px !important;

}
.team-card:hover, .video-card:hover, .about-card:hover  {
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(0, 255, 0, 0.3);
    text-align: center;
}
.team-card img {

    border-radius: 50%;
    padding: 1rem;
    text-align: center;
    border: 3px solid #00ff00;  /* Thicker green border */
    width: 70px !important;
    height: 70px !important;
    margin-bottom: 1.5rem;
}
.target-row-container {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
}

.target-row-image {
    width: 50px;
    height: 50px;
    object-fit: cover;
    border-radius: 5px;
    border: 1px solid #00ff00;
    margin-right: 1rem;
}

.target-row-text {
    color: #ffffff;
    font-size: 1rem;
    text-align: Left;
}
.alert-card {
    background-color: rgba(255, 0, 0, 0.2);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    text-align: center;
    border: 1px solid #ff0000;
}
.alert-card p {
    color: #ffffff;
    font-size: 1rem;
    margin: 0;
}

.chat-message {
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    width: fit-content;
    max-width: 80%;
}

.user-message {
    background-color: rgba(255, 255, 255, 0.1);
    margin-left: auto;
    text-align: right;
}

.bot-message {
    background-color: rgba(0, 255, 0, 0.1);
    margin-right: auto;
    text-align: left;
}

.chat-message p {
    color: #ffffff;
    font-size: 1rem;
    margin: 0;
}



.linkedin-icon {
    width: 30px;
    height: 30px;
    margin-top: 0.5rem;
    transition: transform 0.2s;
    display: inline-block;
}
.linkedin-icon:hover {
    transform: scale(1.2);
}
p, div {
    color: #ffffff;
}
@media (max-width: 768px) {
    .stApp {
        padding-top: 100px;  /* More space for nav bar on mobile */
    }
    .nav-bar {
        flex-direction: column;
        align-items: flex-start;
        padding: 0.5rem;
    }
    .nav-content {
        flex-direction: column;
        align-items: flex-start;
    }
    div.stTabs [data-baseweb="tab-list"] {
        flex-direction: column;
        gap: 0.5rem;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    div.stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0;
    }
    .language-selector {
        width: 100%;
        justify-content: flex-end;
    }
    .language-selector select {
        width: 120px;
    }
    .team-card, .about-card {
        margin: 0.5rem;
        text-align: center;
    }
    .stColumns > div {
        flex: 1 1 100%;
    }
    .hero-section {
        padding: 1rem;
    }
}
</style>
"""
st.markdown(global_css, unsafe_allow_html=True)




    # Sidebar for Language Selector and Alert Settings
with st.sidebar:
    # st.image("SafeCrowd-Picsart-AiImageEnhancer.png", width=100)
    st.image("LogoSafeCrowd.png",width=100)


    # st.image("sda_sda.png",width=90)
    # st.image("lewagon_logo.png" , width=160)  # Consistent width with CSS


    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    col_icon, col_select = st.columns([1, 3])
    with col_icon:
        st.image("https://img.icons8.com/ios-filled/50/ffffff/globe--v1.png", width=24)
    with col_select:
        language = st.selectbox(
            content[st.session_state.language]['nav']['language_label'],
            ["العربية", "English"],
            index=1 if st.session_state.language == 'English' else 0,
            key="language_select",
            label_visibility="collapsed"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.subheader("إعدادات التنبيهات" if st.session_state.language == 'Arabic' else "Alert Settings")
    tawaf_threshold = st.sidebar.number_input(
        "حد الكثافة للطواف" if st.session_state.language == 'Arabic' else "Tawaf Density Threshold",
        min_value=5000, max_value=100000, value=50000, step=1000,
        key="tawaf_threshold"  # Unique key
    )
    saei_threshold = st.sidebar.number_input(
        "حد الكثافة للسعي" if st.session_state.language == 'Arabic' else "Saei Density Threshold",
        min_value=2000, max_value=50000, value=20000, step=500,
        key="saei_threshold"  # Unique key
    )
    other_threshold = st.sidebar.number_input(
        "حد الكثافة للمناطق الأخرى" if st.session_state.language == 'Arabic' else "Other Areas Density Threshold",
        min_value=1000, max_value=20000, value=7800, step=500,
        key="other_threshold"  # Unique key
    )

    # Store thresholds in session state
    st.session_state.alert_thresholds = {
        "Tawaf": tawaf_threshold,
        "Saei": saei_threshold,
        "Other": other_threshold
    }



# Fixed Top Navigation Bar with st.tabs
with st.container():
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    st.markdown('<div class="nav-content">', unsafe_allow_html=True)
    tabs = st.tabs(content[st.session_state.language]['nav']['pages'])
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



# Update language in session state
st.session_state.language = 'Arabic' if language == "العربية" else 'English'



# Team - predict - gallary - resources - faq- about


# Page 1: Team Introduction
with tabs[0]:

    col_1, col_2 = st.columns([3, 1])
    col_2.image("SafeCrowd-Picsart-AiImageEnhancer.png", width=200)
    col_1.image("LogoSafeCrowd.png" ,width=120 )


    # col_1.image("sda_sda.png", width=150)
    # col_2.image("lewagon_logo.png", width=250)

    # Hero Section with Image Background

    pro = content[st.session_state.language]['team']['title']


    # Title and Button
    st.markdown(
        f"""
        <div class="hero-section">
            <h1 class="hero-title">{pro}</h1>


        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Project Overview
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.subheader("نبذة عن المشروع" if st.session_state.language == 'Arabic' else "Project Overview")
    overview = content[st.session_state.language]['team']['overview']
    text_align = "right" if st.session_state.language == 'Arabic' else "left"
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div>
            <h4 style="text-align: justify; direction: {'rtl' if st.session_state.language == 'Arabic' else 'ltr'}; line-height: 1.6;">
                {overview}
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 3. Target Audience
    st.subheader("الفئة المستهدفة" if st.session_state.language == 'Arabic' else "Target Audience")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("team.png", width=150)

    with col2:
        for i, target in enumerate(content[st.session_state.language]['team']['target_audience']):
            teams = target['target']
            img = target['img']
            benefit = target['benefit']
            st.markdown(
                f"""
                <div class="target-row-container" style="text-align: {'right' if st.session_state.language == 'Arabic' else 'left'}; direction: {'rtl' if st.session_state.language == 'Arabic' else 'ltr'};">
                    <img src="{img}" class="target-row-image">
                    <h5 class="target-row-text">{teams}</h5>
                </div>
                <p style="text-align: {'right' if st.session_state.language == 'Arabic' else 'left'}; direction: {'rtl' if st.session_state.language == 'Arabic' else 'ltr'}; margin-right: {'1rem' if st.session_state.language == 'Arabic' else '0'}; margin-left: {'0' if st.session_state.language == 'Arabic' else '1rem'};">
                    {benefit}
                </p>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.divider()

    # 4. Expected Results
    st.subheader("النتائج المتوقعة" if st.session_state.language == 'Arabic' else "Expected Results")
    coll = st.columns(3)

    for i, result in enumerate(content[st.session_state.language]['team']['expected_results']):
        with coll[i]:
            goal = result['result']

            st.markdown(
                f"""
                <div class="about-card">
                    <h4 style="font-size: 1.3rem;">{goal}</h4>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 5. Makkah Crowd Management Zones Map
    st.subheader(content[st.session_state.language]['team']['map_title'])
    st.write(" تعرض الخريطة المناطق الرئيسية المراقبة باستخدام نماذج التعلم العميق لتحليل كثافة الحشود في الوقت الفعلي. وتشمل التالي : الطواف - الصفا - اخرى" if st.session_state.language == 'Arabic' else "The map shows key areas monitored using deep learning models for real-time crowd density analysis. including : Tawaf  , Safa , Other")
    df_map = pd.DataFrame({
        "lat": [21.4225, 21.4200, 21.4250, 21.4225],  # Repeated Kaaba for Tawaf
        "lon": [39.8262, 39.8250, 39.8270, 39.8262],
        "location": content[st.session_state.language]['team']['map_locations']
    })
    fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", hover_name="location", zoom=15, color_discrete_sequence=["#00ff00"])
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 6. Team Members
    st.header(content[st.session_state.language]['team']['team_title'])
    cols = st.columns(5)

    for i, member in enumerate(content[st.session_state.language]['team']['team']):
        with cols[i]:
            name = member['name']
            more = member['more']
            linkedin_url = member['linkedin']
            st.markdown(
                f"""
                <div class="team-card">
                    <h5 style="font-size: 1rem;">{name}</h5>
                    <p>{more}</p>
                    <a href="{linkedin_url}"><img src="https://img.icons8.com/color/30/000000/linkedin.png" class="linkedin-icon"></a>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Page 3: Crowd Prediction
with tabs[1]:
    # Title
    st.title(content[st.session_state.language]['prediction']['title'])
    st.write(content[st.session_state.language]['prediction']['info'])

    # Predict Button
    if st.button(content[st.session_state.language]['prediction']['predict_button'], key="predict_button"):
        with st.spinner("جارٍ إنشاء التوقع..." if st.session_state.language == 'Arabic' else "Generating forecast..."):
            try:
                # Generate forecast using real.py
                forecast_df = generate_forecast(language=st.session_state.language)

                # Store forecast in session state
                st.session_state.forecast_df = forecast_df

                # Display Forecast Results
                forecast_container = st.container(border=True)
                with forecast_container:
                    st.subheader(content[st.session_state.language]['prediction']['forecast_title'].format(days=7))
                    # st.write(content[st.session_state.language]['prediction']['prediction_time'].format(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                    # Prepare display DataFrame
                    display_df = forecast_df[['Date', 'Total_Predicted', 'Tawaf', 'Saei', 'Other', 'Crowd_Level', 'Tawaf_Crowd_Level', 'Saei_Crowd_Level', 'Other_Crowd_Level']].copy()
                    display_df['Total'] = display_df['Total_Predicted'].astype(str) + " (" + display_df['Crowd_Level'] + ")"
                    display_df['Tawaf'] = display_df['Tawaf'].astype(str) + " (" + display_df['Tawaf_Crowd_Level'] + ")"
                    display_df['Saei'] = display_df['Saei'].astype(str) + " (" + display_df['Saei_Crowd_Level'] + ")"
                    display_df['Other'] = display_df['Other'].astype(str) + " (" + display_df['Other_Crowd_Level'] + ")"
                    display_df = display_df[['Date', 'Total', 'Tawaf', 'Saei', 'Other']]

                    # Color-code cells
                    def color_cells(val):
                        if "مرتفع" in val or "High" in val:
                            return "background-color: red"
                        elif "متوسط" in val or "Medium" in val:
                            return "background-color: orange"
                        else:
                            return "background-color: green"
                    st.dataframe(
                        display_df.style.applymap(color_cells, subset=['Total', 'Tawaf', 'Saei', 'Other']),
                        hide_index=True,
                        use_container_width=True
                    )

                    # Alerts
                    st.subheader("التنبيهات" if st.session_state.language == 'Arabic' else "Alerts")
                    thresholds = st.session_state.alert_thresholds
                    alerts = []

                    high_tawaf = forecast_df[forecast_df['Tawaf'] > thresholds['Tawaf']]
                    if not high_tawaf.empty:
                        alerts.append(
                            "⚠ كثافة مرتفعة في الطواف في الأيام التالية: " + ", ".join(high_tawaf['Date'].astype(str))
                            if st.session_state.language == 'Arabic'
                            else "⚠ High density in Tawaf on the following days: " + ", ".join(high_tawaf['Date'].astype(str))
                        )

                    high_saei = forecast_df[forecast_df['Saei'] > thresholds['Saei']]
                    if not high_saei.empty:
                        alerts.append(
                            "⚠ كثافة مرتفعة في السعي في الأيام التالية: " + ", ".join(high_saei['Date'].astype(str))
                            if st.session_state.language == 'Arabic'
                            else "⚠ High density in Saei on the following days: " + ", ".join(high_saei['Date'].astype(str))
                        )

                    high_other = forecast_df[forecast_df['Other'] > thresholds['Other']]
                    if not high_other.empty:
                        alerts.append(
                            "⚠ كثافة مرتفعة في المناطق الأخرى في الأيام التالية: " + ", ".join(high_other['Date'].astype(str))
                            if st.session_state.language == 'Arabic'
                            else "⚠ High density in Other areas on the following days: " + ", ".join(high_other['Date'].astype(str))
                        )

                    if alerts:
                        for alert in alerts:
                            st.markdown(
                                f"""
                                <div class="alert-card">
                                    <p>{alert}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("لا توجد تنبيهات حالياً بناءً على الإعدادات." if st.session_state.language == 'Arabic' else "No alerts based on current settings.")

                    # Plotly bar chart
                    forecast_df['Average Crowd Size'] = forecast_df[['Tawaf', 'Saei', 'Other']].mean(axis=1)
                    fig = px.bar(
                        forecast_df,
                        x='Date',
                        y='Average Crowd Size',
                        title="توقع الحشود" if st.session_state.language == 'Arabic' else "Crowd Forecast",
                        labels={"Average Crowd Size": "متوسط حجم الحشود" if st.session_state.language == 'Arabic' else "Average Crowd Size"}
                    )

                    fig.update_traces(marker_color='#00ff00')
                    fig.update_layout(
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        font_color='white',
                        title_font_color='white'

                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(content[st.session_state.language]['prediction']['error'].format(error=str(e)))

    # Crowd Distribution Map
    if 'forecast_df' in st.session_state and st.session_state.forecast_df is not None:
        st.subheader(content[st.session_state.language]['prediction']['map_title'])
        selected_day = st.selectbox(
            content[st.session_state.language]['prediction']['select_day'],
            st.session_state.forecast_df['Date'],
            index=0
        )

        # Prepare map data for selected day
        selected_row = st.session_state.forecast_df[st.session_state.forecast_df['Date'] == selected_day]
        if not selected_row.empty:
            locations = pd.DataFrame({
                "lat": [21.4225, 21.4200, 21.4230],  # Approximate coordinates for Tawaf, Saei, Other
                "lon": [39.8262, 39.8250, 39.8260],
                "location": ["الطواف", "السعي", "أخرى"] if st.session_state.language == 'Arabic' else ["Tawaf", "Saei", "Other"],
                "crowd_size": [
                    selected_row['Tawaf'].iloc[0],
                    selected_row['Saei'].iloc[0],
                    selected_row['Other'].iloc[0]
                ],
                "crowd_level": [
                    selected_row['Tawaf_Crowd_Level'].iloc[0],
                    selected_row['Saei_Crowd_Level'].iloc[0],
                    selected_row['Other_Crowd_Level'].iloc[0]
                ]
            })
            locations['size'] = locations['crowd_size'] / 1000  # Scale for visibility

            # Plotly scatter map
            fig = px.scatter_mapbox(
                locations,
                lat="lat",
                lon="lon",
                hover_name="location",
                hover_data={"crowd_size": True, "crowd_level": True},
                size="size",
                color="crowd_level",
                color_discrete_map={
                    "منخفض": "green", "متوسط": "orange", "مرتفع": "red",
                    "Low": "green", "Medium": "orange", "High": "red"
                },
                zoom=17
            )
            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("لم يتم العثور على بيانات التوقع لليوم المحدد." if st.session_state.language == 'Arabic' else "No forecast data found for the selected day.")


# Page 4: About
with tabs[5]:

    # Title
    container = st.container(border=True)
    with container:
        st.title(content[st.session_state.language]['about']['title'])
        st.write("")
        st.markdown('</div>', unsafe_allow_html=True)

    # Vision Section
    st.subheader(content[st.session_state.language]['about']['vision'])
    st.write(content[st.session_state.language]['about']['vision_text'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Mission Section
    st.subheader(content[st.session_state.language]['about']['mission'])
    st.write(content[st.session_state.language]['about']['mission_text'])
    st.markdown('</div>', unsafe_allow_html=True)
# Contact Section
    st.subheader(content[st.session_state.language]['about']['chatbot'])
    st.markdown('</div>', unsafe_allow_html=True)

# Center QR code and Telegram link
    st.markdown(
    """
    <style>
    .centered-content {

        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 2rem auto;  /* Increased margin for better spacing */
        width: 100%;  /* Ensure it takes full container width */
        max-width: 600px;  /* Limit max width for larger screens */
    }
    .centered-content img {
        max-width: 150px;  /* Smaller size for balance */
        width: 100%;  /* Responsive width */
        margin-bottom: 1.5rem;  /* Space between image and link */
        شlign-items: center;
        justify-content: center;
        text-align: center;
    }
    .centered-content a {
        color: #00ff00;  /* Green color for link */
        font-size: 1.1rem;
        text-decoration: none;
        word-break: break-all;  /* Prevent link overflow */
    }
    .centered-content a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.image("https://i.imgur.com/1nAbAwj.png", width=150)  # Consistent width with CSS
    st.markdown(
    '<a href="https://t.me/HaramGuideBot">https://t.me/HaramGuideBot</a>',
    unsafe_allow_html=True
)
    st.markdown('</div>', unsafe_allow_html=True)


# Tab 4: FAQ and Chatbot Support
with tabs[4]:
    st.title("الأسئلة الشائعة والدعم" if st.session_state.language == 'Arabic' else "FAQ and Support")

    # FAQ Section
    st.subheader("الأسئلة الشائعة" if st.session_state.language == 'Arabic' else "Frequently Asked Questions")
    faqs = {
        "كيف يتم إنشاء توقعات الحشود؟" if st.session_state.language == 'Arabic' else "How are crowd forecasts generated?":
            "نستخدم نماذج الذكاء الاصطناعي لتحليل البيانات التاريخية والبيانات في الوقت الفعلي لتوقع كثافة الحشود." if st.session_state.language == 'Arabic'
            else "We use AI models to analyze historical and real-time data to predict crowd density.",
        "من هم الفئات المستهدفة للمشروع؟" if st.session_state.language == 'Arabic' else "Who is the target audience for this project?":
            "إدارة الحرم المكي، السلطات الأمنية، وفرق الطوارئ." if st.session_state.language == 'Arabic'
            else "Makkah Haram Administration, Security Authorities, and Emergency Response Teams.",
        "كيف يمكنني تخصيص التنبيهات؟" if st.session_state.language == 'Arabic' else "How can I customize alerts?":
            "استخدم إعدادات التنبيهات في الشريط الجانبي لتحديد حدود الكثافة لكل منطقة." if st.session_state.language == 'Arabic'
            else "Use the Alert Settings in the sidebar to set density thresholds for each area.",
        "كيفية استخدام تشات بوت تلقرام؟" if st.session_state.language == 'Arabic' else "How to use Telegram chatbot?":
            "" if st.session_state.language == 'Arabic'
            else "By scanning the QR code on the project page or clicking on the link"
    }

    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)

    # Chatbot Section
    st.subheader("دعم الدردشة" if st.session_state.language == 'Arabic' else "Chat Support")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input(
        "اسأل سؤالاً..." if st.session_state.language == 'Arabic' else "Ask a question...",
        key="chat_input"
    )

    if user_input:
        # Rule-based responses
        user_input_lower = user_input.lower()
        response = "آسف، لم أفهم سؤالك. حاول صياغته بشكل مختلف أو راجع الأسئلة الشائعة." if st.session_state.language == 'Arabic' else "Sorry, I didn’t understand your question. Try rephrasing or check the FAQ."

        if "توقعات" in user_input_lower or "forecast" in user_input_lower:
            response = "لإنشاء توقعات، انتقل إلى صفحة التوقعات واضغط على زر 'تنبؤ'. يمكنك بعدها رؤية الجدول والرسوم البيانية." if st.session_state.language == 'Arabic' else "To generate a forecast, go to the Prediction page and click the 'Predict' button. You can then view the table and charts."
        elif "تنبيهات" in user_input_lower or "alerts" in user_input_lower:
            response = "يمكنك تخصيص التنبيهات من الشريط الجانبي بتحديد حدود الكثافة لكل منطقة." if st.session_state.language == 'Arabic' else "You can customize alerts from the sidebar by setting density thresholds for each area."
        elif "خريطة" in user_input_lower or "map" in user_input_lower:
            response = "خريطة توزيع الحشود متاحة في صفحة التوقعات بعد إنشاء التوقع. اختر يوماً لعرض التوزيع." if st.session_state.language == 'Arabic' else "The crowd distribution map is available on the Prediction page after generating a forecast. Select a day to view the distribution."

        # Add to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="chat-message user-message">
                <p>{chat['user']}</p>
            </div>
            <div class="chat-message bot-message">
                <p>{chat['bot']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Tab 5: Resource Allocation Planner
with tabs[3]:


    st.title("تخطيط توزيع الموارد" if st.session_state.language == 'Arabic' else "Resource Allocation Planner")

    if 'forecast_df' in st.session_state and st.session_state.forecast_df is not None:
        with st.form("resource_form"):
            security_ratio = st.slider(
                "نسبة أفراد الأمن لكل شخص" if st.session_state.language == 'Arabic' else "Security Personnel Available",
                100, 5000, 2000, step=100,
                key="security_ratio"  # Unique key
            )
            medical_ratio = st.slider(
                "نسبة الفرق الطبية لكل شخص" if st.session_state.language == 'Arabic' else "Medical Teams Available",
                100, 5000, 2000, step=100,
                key="medical_ratio"  # Unique key
            )
            selected_day = st.selectbox(
                "اختر يوم التخطيط" if st.session_state.language == 'Arabic' else "Select Planning Day",
                st.session_state.forecast_df["Date"],
                key="resource_day_select"  # Unique key
            )
            submitted = st.form_submit_button("حساب الموارد" if st.session_state.language == 'Arabic' else "Calculate Resources")

            if submitted:
                # Get crowd sizes for the selected day
                selected_row = st.session_state.forecast_df[st.session_state.forecast_df["Date"] == selected_day]
                tawaf_crowd = selected_row["Tawaf"].iloc[0]
                saei_crowd = selected_row["Saei"].iloc[0]
                other_crowd = selected_row["Other"].iloc[0]

                # Calculate resource needs
                resources = pd.DataFrame({
                    "Location": ["الطواف", "السعي", "أخرى"] if st.session_state.language == 'Arabic' else ["Tawaf", "Saei", "Other"],
                    "Crowd Size": [tawaf_crowd, saei_crowd, other_crowd],
                    "Security Personnel Ratio": [int(crowd / security_ratio) for crowd in [tawaf_crowd, saei_crowd, other_crowd]],
                    "Medical Teams Ratio": [int(crowd / medical_ratio) for crowd in [tawaf_crowd, saei_crowd, other_crowd]]
                })

                # Display results
                st.subheader("الموارد المطلوبة" if st.session_state.language == 'Arabic' else "Required Resources")
                st.dataframe(resources, use_container_width=True)

                # Bar chart for resource distribution
                # fig = px.bar(
                #     resources,
                #     x=["Location"],
                #     y="Security Personnel Ratio",
                #     title="توزيع الموارد حسب الموقع" if st.session_state.language == 'Arabic' else "Resource Distribution by Location",
                #     labels={
                #         "value": "العدد" if st.session_state.language == 'Arabic' else "Count",
                #         "variable": "نوع المورد" if st.session_state.language == 'Arabic' else "Resource Type"
                #     }
                # )
                # fig.update_traces(marker_color='#00ff00')
                # fig.update_layout(
                #     plot_bgcolor='black',
                #     paper_bgcolor='black',
                #     font_color='white',
                #     title_font_color='white'
                # )
                # st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("يرجى إنشاء توقعات أولاً من صفحة التوقعات." if st.session_state.language == 'Arabic' else "Please generate a forecast first from the Prediction page.")
# Page 6: Video Gallery & Live Detection
with tabs[2]:
    st.title(content[st.session_state.language]['video']['title'])
    st.write(content[st.session_state.language]['video']['description'])

    # Video Gallery
    st.subheader("مقاطع فيديو إدارة الحشود" if st.session_state.language == 'Arabic' else "Crowd Management Videos")
    col1, col2 = st.columns(2)
    video_files = [
        "videos/output_result_tawaf_compresesed.mp4",
        # "Footage1.mp4",
        "videos/output_result_tawaf2_compressed.mp4",
        "videos/output_result_saye_compressed.mp4",
        "videos/output_result_kabbah_compressed.mp4"  # Will be labeled as "Other Areas"
    ]
    captions = content[st.session_state.language]['video']['captions']

    for i, (video_file, caption) in enumerate(zip(video_files, captions)):
        try:
            with col1 if i % 2 == 0 else col2:
                st.markdown(
                    f'<div style="text-align: center; color: #00ff00;"><h4>{caption}</h4></div>',
                    unsafe_allow_html=True
                )
                st.video(video_file)
        except FileNotFoundError:
            st.error(content[st.session_state.language]['video']['error'].format(file=video_file))

    # Divider
    st.markdown("---")

    # Real-Time Detection Section
    st.subheader(content[st.session_state.language]['realtime']['title'])
    st.write(content[st.session_state.language]['realtime']['description'])

    # Sidebar for Detection Settings
    with st.sidebar:
        st.header("إعدادات الكشف" if st.session_state.language == 'Arabic' else "Detection Settings")
        confidence_threshold = st.slider(
            "عتبة الثقة" if st.session_state.language == 'Arabic' else "Confidence Threshold",
            0.0, 1.0, 0.5, 0.01,
            key="confidence_threshold"
        )
        webcam_index = st.number_input(
            "مؤشر كاميرا الويب" if st.session_state.language == 'Arabic' else "Webcam Index",
            0, 4, 0,
            key="webcam_index"
        )
        crop_percentage = st.slider(
            "نسبة قص الجزء العلوي (لاستبعاد السقف)" if st.session_state.language == 'Arabic' else "Crop Top Percentage (to exclude ceiling)",
            0.0, 0.5, 0.3, 0.01,
            key="crop_percentage"
        )
        person_count_threshold = st.number_input(
            "حد عدد الأشخاص للتنبيه" if st.session_state.language == 'Arabic' else "Person Count Threshold for Alert",
            1, 100, 10,
            key="person_count_threshold"
        )
        coverage_threshold = st.slider(
            "حد نسبة التغطية للتنبيه" if st.session_state.language == 'Arabic' else "Coverage Percentage Threshold for Alert",
            0.0, 1.0, 0.5, 0.01,
            key="coverage_threshold"
        )

    # Load YOLOv8s-seg model
    @st.cache_resource
    def load_model():
        return YOLO("yolov8s-seg.pt")
    model = load_model()

    # Webcam initialization function
    def get_webcam_feed(index=0):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        return None

    # Initialize session state
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'last_webcam_index' not in st.session_state:
        st.session_state.last_webcam_index = webcam_index

    # Reinitialize webcam if index changes
    if st.session_state.last_webcam_index != webcam_index:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.last_webcam_index = webcam_index
        st.session_state.cap = get_webcam_feed(webcam_index)

    # Initialize webcam if not already done
    if st.session_state.cap is None:
        st.session_state.cap = get_webcam_feed(webcam_index)

    # Webcam Detection
    video_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button(content[st.session_state.language]['realtime']['start_button'], key="start_button_realtime")
    with col2:
        stop_button = st.button(content[st.session_state.language]['realtime']['stop_button'], key="stop_button_realtime")

    # Start detection
    if start_button and not st.session_state.detection_running:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.session_state.cap = get_webcam_feed(webcam_index)
        if st.session_state.cap is None:
            st.error(content[st.session_state.language]['realtime']['error'])
        else:
            st.session_state.detection_running = True

    # Stop detection
    if stop_button and st.session_state.detection_running:
        st.session_state.detection_running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        cv2.destroyAllWindows()
        video_placeholder.empty()

    # Run webcam detection loop
    if st.session_state.detection_running and st.session_state.cap is not None and st.session_state.cap.isOpened():
        frame_count = 0
        while st.session_state.detection_running and st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning(content[st.session_state.language]['realtime']['warning'])
                st.session_state.detection_running = False
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue  # Skip every other frame to reduce lag

            # Resize frame
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crop the bottom for calculations
            height, width = frame_rgb.shape[:2]
            roi_top = int(height * crop_percentage)
            roi_frame = frame_rgb[roi_top:, :]
            roi_height, roi_width = roi_frame.shape[:2]

            # Predict using YOLOv8s-seg
            results = model.predict(frame_rgb, conf=confidence_threshold)
            result = results[0]

            # Filter for person class (class 0)
            person_indices = (result.boxes.cls.cpu().numpy() == 0)
            filtered_boxes = result.boxes[person_indices]
            filtered_masks = result.masks.data[person_indices] if result.masks is not None else []

            # Count people
            person_count = len(filtered_boxes)

            # Combine all person masks
            mask_total = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
            if len(filtered_masks) > 0:
                for m in filtered_masks:
                    mask = m.cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    mask_resized = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]))
                    mask_total = cv2.bitwise_or(mask_total, mask_resized)

            # Crop the mask to ROI
            mask_roi = mask_total[roi_top:, :]

            # Calculate coverage percentage
            person_pixels = np.sum(mask_roi)
            total_pixels = mask_roi.shape[0] * mask_roi.shape[1]
            person_ratio = person_pixels / total_pixels if total_pixels > 0 else 0

            # Annotate frame
            annotated_frame = frame_rgb.copy()
            if np.sum(mask_total) > 0:
                empty_mask = 1 - mask_total
                color_mask = np.zeros_like(annotated_frame)
                color_mask[roi_top:, :, 0] = empty_mask[roi_top:, :] * 255  # Blue overlay
                annotated_frame = cv2.addWeighted(annotated_frame, 1, color_mask, 0.4, 0)

            # Draw person boxes
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green boxes

            # Display People count and Crowded percentage
            cv2.putText(
                annotated_frame,
                f"{'الأشخاص' if st.session_state.language == 'Arabic' else 'PEOPLE'}: {person_count} | {'الازدحام' if st.session_state.language == 'Arabic' else 'CROWDED'}: {person_ratio:.1%}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),  # Red text
                2,
                cv2.LINE_AA
            )

            # Show output
            video_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

            # Alerts
            if person_count > person_count_threshold:
                st.markdown(
                    f"""
                    <div class="alert-card">
                        <p>{"⚠ عدد الأشخاص يتجاوز الحد: " if st.session_state.language == 'Arabic' else "⚠ Person count exceeds threshold: "} {person_count}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if person_ratio > coverage_threshold:
                st.markdown(
                    f"""
                    <div class="alert-card">
                        <p>{"⚠ نسبة التغطية تتجاوز الحد: " if st.session_state.language == 'Arabic' else "⚠ Coverage percentage exceeds threshold: "} {person_ratio:.1%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    elif st.session_state.detection_running and (st.session_state.cap is None or not st.session_state.cap.isOpened()):
        st.error(content[st.session_state.language]['realtime']['error'])
        st.session_state.detection_running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

    # Video File Upload Section
    st.subheader("تحميل فيديو للكشف" if st.session_state.language == 'Arabic' else "Upload Video for Detection")
    st.info("بدلاً من ذلك، يمكنك تحميل ملف فيديو:" if st.session_state.language == 'Arabic' else "As an alternative, you can upload a video file:")
    uploaded_file = st.file_uploader(
        "اختر ملف فيديو" if st.session_state.language == 'Arabic' else "Choose a video file",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    if uploaded_file is not None:
        try:
            # Save to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()

            # Process video file
            cap = cv2.VideoCapture(tfile.name)
            video_placeholder = st.empty()
            stop_upload_button = st.button(
                "إيقاف تشغيل الفيديو" if st.session_state.language == 'Arabic' else "Stop Video Playback",
                key="stop_upload_video"
            )
            frame_count = 0
            stop_flag = False

            while cap.isOpened() and not stop_flag:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 2 != 0:
                    continue  # Skip every other frame to reduce lag

                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Crop the bottom for calculations
                height, width = frame_rgb.shape[:2]
                roi_top = int(height * 0.50)
                roi_frame = frame_rgb[roi_top:, :]
                roi_height, roi_width = roi_frame.shape[:2]

                # Predict using YOLOv8s-seg
                results = model.predict(frame_rgb, conf=confidence_threshold)
                result = results[0]

                # Filter for person class (class 0)
                person_indices = (result.boxes.cls.cpu().numpy() == 0)
                filtered_boxes = result.boxes[person_indices]
                filtered_masks = result.masks.data[person_indices] if result.masks is not None else []

                # Count people
                person_count = len(filtered_boxes)

                # Combine all person masks
                mask_total = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
                if len(filtered_masks) > 0:
                    for m in filtered_masks:
                        mask = m.cpu().numpy()
                        mask = (mask > 0.5).astype(np.uint8)
                        mask_resized = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]))
                        mask_total = cv2.bitwise_or(mask_total, mask_resized)

                # Crop the mask to ROI
                mask_roi = mask_total[roi_top:, :]

                # Calculate coverage percentage
                person_pixels = np.sum(mask_roi)
                total_pixels = mask_roi.shape[0] * mask_roi.shape[1]
                person_ratio = person_pixels / total_pixels if total_pixels > 0 else 0

                # Annotate frame
                annotated_frame = frame_rgb.copy()
                if np.sum(mask_total) > 0:
                    empty_mask = 1 - mask_total
                    color_mask = np.zeros_like(annotated_frame)
                    color_mask[roi_top:, :, 0] = empty_mask[roi_top:, :] * 255  # Blue overlay
                    annotated_frame = cv2.addWeighted(annotated_frame, 1, color_mask, 0.4, 0)

                # Draw person boxes
                for box in filtered_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green boxes

                # Display People count and Crowded percentage
                cv2.putText(
                    annotated_frame,
                    f"{'الأشخاص' if st.session_state.language == 'Arabic' else 'PEOPLE'}: {person_count} | {'الازدحام' if st.session_state.language == 'Arabic' else 'CROWDED'}: {person_ratio:.1%}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),  # Red text
                    2,
                    cv2.LINE_AA
                )

                # Show output
                video_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

                # Alerts
                if person_count > person_count_threshold:
                    st.markdown(
                        f"""
                        <div class="alert-card">
                            <p>{"⚠ عدد الأشخاص يتجاوز الحد: " if st.session_state.language == 'Arabic' else "⚠ Person count exceeds threshold: "} {person_count}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                if person_ratio > coverage_threshold:
                    st.markdown(
                        f"""
                        <div class="alert-card">
                            <p>{"⚠ نسبة التغطية تتجاوز الحد: " if st.session_state.language == 'Arabic' else "⚠ Coverage percentage exceeds threshold: "} {person_ratio:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if stop_upload_button:
                    stop_flag = True
                    break

            cap.release()
            st.success("تمت معالجة الفيديو بنجاح!" if st.session_state.language == 'Arabic' else "Video processed successfully!")

        except Exception as e:
            st.error(f"خطأ في معالجة الفيديو: {str(e)}" if st.session_state.language == 'Arabic' else f"Error processing video: {str(e)}")


# ---------------------------

# تبويب إدارة الأدوار
with tabs[6]:

    st.title("إدارة الأدوار" if st.session_state.language == 'Arabic' else "Floor Management")
    st.write("مسح رموز QR لإدارة دخول المعتمرين إلى الأدوار الثلاثة." if st.session_state.language == 'Arabic' else "Scan QR codes to manage pilgrim entry to the three floors.")

    # التحقق من وجود جدول floors
    try:
        conn = sqlite3.connect("crowd_management.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='floors'")
        if not cursor.fetchone():
            st.error("جدول 'floors' غير موجود. يرجى إعادة تشغيل التطبيق أو التحقق من قاعدة البيانات.")
            logger.error("Table 'floors' does not exist in the database.")
            init_db()  # محاولة إعادة إنشاء الجدول
        else:
            cursor.execute("SELECT floor, current_count, max_capacity, status FROM floors")
            floors_data = pd.DataFrame(cursor.fetchall(), columns=["Floor", "Current Count", "Max Capacity", "Status"])
            if st.session_state.language == 'Arabic':
                floors_data["Floor"] = floors_data["Floor"].replace({"Ground": "الأرضي", "First": "الأول", "Second": "الثاني"})
                floors_data["Status"] = floors_data["Status"].replace({"Available": "متاح", "Crowded": "مزدحم"})

            st.subheader("حالة الأدوار" if st.session_state.language == 'Arabic' else "Floors Status")
            st.dataframe(floors_data, use_container_width=True)

            # رسم بياني
            fig = px.bar(
                floors_data,
                x="Floor",
                y="Current Count",
                color="Status",
                title="عدد الأشخاص في كل دور" if st.session_state.language == 'Arabic' else "Number of People per Floor",
                color_discrete_map={"متاح": "green", "مزدحم": "red", "Available": "green", "Crowded": "red"}
            )
            fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)

            # إنشاء رمز QR
            st.subheader("إنشاء رمز QR لمعتمر جديد" if st.session_state.language == 'Arabic' else "Generate QR Code for New Pilgrim")
            with st.form("generate_qr_form"):
                pilgrim_id = str(uuid.uuid4())
                if st.form_submit_button("إنشاء" if st.session_state.language == 'Arabic' else "Generate"):
                    qr_data = {
                        "pilgrim_id": pilgrim_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    encrypted_data = cipher.encrypt(json.dumps(qr_data).encode()).decode()
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(encrypted_data)
                    qr.make(fit=True)
                    img = qr.make_image(fill="black", back_color="white")
                    qr_path = f"qr_codes/{pilgrim_id}.png"
                    img.save(qr_path)
                    st.image(qr_path, caption="رمز QR للمعتمر" if st.session_state.language == 'Arabic' else "QR Code for Pilgrim")
                    st.write(f"معرف المعتمر: {pilgrim_id}")
                    st.write(f"محتوى رمز QR (للمحاكاة): {encrypted_data}")

            # مسح رمز QR
            st.subheader("مسح رمز QR" if st.session_state.language == 'Arabic' else "Scan QR Code")
            with st.form("scan_qr_form"):
                qr_content = st.text_input(
                    "أدخل محتوى رمز QR (للمحاكاة)" if st.session_state.language == 'Arabic' else "Enter QR Code Content (for simulation)",
                    help="انسخ النص المشفر من رمز QR المُنشأ أو امسح رمز QR باستخدام قارئ" if st.session_state.language == 'Arabic' else "Copy the encrypted text from the generated QR code or scan the QR code with a reader"
                )
                selected_floor = st.selectbox(
                    "اختر الدور" if st.session_state.language == 'Arabic' else "Select Floor",
                    ["الأرضي", "الأول", "الثاني"] if st.session_state.language == 'Arabic' else ["Ground", "First", "Second"],
                    key="qr_floor_select"
                )
                scan_type = st.radio(
                    "نوع المسح" if st.session_state.language == 'Arabic' else "Scan Type",
                    ["دخول", "خروج"] if st.session_state.language == 'Arabic' else ["Entry", "Exit"],
                    key="qr_scan_type"
                )
                submitted = st.form_submit_button("مسح" if st.session_state.language == 'Arabic' else "Scan")

                if submitted:
                    if not qr_content:
                        st.error("يرجى إدخال محتوى رمز QR." if st.session_state.language == 'Arabic' else "Please enter QR code content.")
                    else:
                        try:
                            # فك التشفير
                            decrypted_data = json.loads(cipher.decrypt(qr_content.encode()).decode())
                            pilgrim_id = decrypted_data.get("pilgrim_id")
                            if not pilgrim_id:
                                raise ValueError("معرف المعتمر مفقود في بيانات رمز QR.")

                            # تحويل اسم الدور إلى اسم قاعدة البيانات
                            floor_map = {"الأرضي": "Ground", "الأول": "First", "الثاني": "Second"} if st.session_state.language == 'Arabic' else {}
                            db_floor = floor_map.get(selected_floor, selected_floor)

                            # التحقق من اتصال قاعدة البيانات
                            cursor.execute("SELECT 1 FROM floors WHERE floor = ?", (db_floor,))
                            if not cursor.fetchone():
                                raise sqlite3.Error(f"الدور {db_floor} غير موجود في قاعدة البيانات.")

                            if scan_type == "دخول" or scan_type == "Entry":
                                cursor.execute("SELECT current_count, max_capacity FROM floors WHERE floor = ?", (db_floor,))
                                result = cursor.fetchone()
                                if not result:
                                    raise sqlite3.Error(f"لا يمكن استرجاع بيانات الدور {db_floor}.")
                                current_count, max_capacity = result

                                if current_count >= max_capacity * 0.95:  # تحذير عند اقتراب السعة
                                    st.warning(f"الدور {selected_floor} يقترب من السعة القصوى ({current_count}/{max_capacity}).")
                                if current_count < max_capacity:
                                    cursor.execute(
                                        "INSERT OR REPLACE INTO pilgrims (id, floor, entry_time) VALUES (?, ?, ?)",
                                        (pilgrim_id, db_floor, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    )
                                    cursor.execute(
                                        "UPDATE floors SET current_count = current_count + 1, status = ?, last_updated = ? WHERE floor = ?",
                                        ("Crowded" if current_count + 1 >= max_capacity else "Available",
                                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), db_floor)
                                    )
                                    cursor.execute(
                                        "INSERT INTO activity_log (pilgrim_id, action, floor, timestamp) VALUES (?, ?, ?, ?)",
                                        (pilgrim_id, "Entry", db_floor, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    )
                                    conn.commit()
                                    st.success(f"تم فتح البوابة للدور {selected_floor}!" if st.session_state.language == 'Arabic' else f"Gate opened for {selected_floor} floor!")
                                else:
                                    cursor.execute("SELECT floor FROM floors WHERE current_count < max_capacity AND floor != ?", (db_floor,))
                                    available_floors = [row[0] for row in cursor.fetchall()]
                                    if st.session_state.language == 'Arabic':
                                        available_floors = [k for k, v in floor_map.items() if v in available_floors]
                                    if available_floors:
                                        st.warning(f"الدور {selected_floor} مزدحم، جرب: {', '.join(available_floors)}")
                                    else:
                                        st.error("جميع الأدوار مزدحمة، يرجى المحاولة لاحقًا.")
                            else:  # خروج
                                cursor.execute(
                                    "UPDATE pilgrims SET exit_time = ? WHERE id = ? AND floor = ? AND exit_time IS NULL",
                                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pilgrim_id, db_floor)
                                )
                                if cursor.rowcount > 0:
                                    cursor.execute(
                                        "UPDATE floors SET current_count = current_count - 1, status = 'Available', last_updated = ? WHERE floor = ?",
                                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), db_floor)
                                    )
                                    cursor.execute(
                                        "INSERT INTO activity_log (pilgrim_id, action, floor, timestamp) VALUES (?, ?, ?, ?)",
                                        (pilgrim_id, "Exit", db_floor, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    )
                                    conn.commit()
                                    st.success(f"تم تسجيل الخروج من الدور {selected_floor}!" if st.session_state.language == 'Arabic' else f"Exit recorded from {selected_floor} floor!")
                                else:
                                    st.error("لم يتم العثور على سجل دخول لهذا المعتمر في هذا الدور." if st.session_state.language == 'Arabic' else "No entry record found for this pilgrim on this floor.")
                        except cryptography.fernet.InvalidToken:
                            st.error("رمز QR غير صالح أو مفتاح التشفير غير متطابق. تأكد من استخدام نفس المفتاح." if st.session_state.language == 'Arabic' else "Invalid QR code or encryption key mismatch. Ensure the same key is used.")
                            logger.error("InvalidToken: Failed to decrypt QR code content.")
                        except json.JSONDecodeError:
                            st.error("بيانات رمز QR غير صالحة. تأكد من إدخال محتوى مشفر صحيح." if st.session_state.language == 'Arabic' else "Invalid QR code data. Ensure correct encrypted content is entered.")
                            logger.error("JSONDecodeError: Failed to parse decrypted QR code data.")
                        except ValueError as ve:
                            st.error(f"خطأ في بيانات رمز QR: {str(ve)}" if st.session_state.language == 'Arabic' else f"QR code data error: {str(ve)}")
                            logger.error(f"ValueError: {str(ve)}")
                        except sqlite3.Error as dbe:
                            st.error(f"خطأ في قاعدة البيانات: {str(dbe)}" if st.session_state.language == 'Arabic' else f"Database error: {str(dbe)}")
                            logger.error(f"Database error: {str(dbe)}")
                        except Exception as e:
                            st.error(f"خطأ غير متوقع في معالجة رمز QR: {str(e)}" if st.session_state.language == 'Arabic' else f"Unexpected error processing QR code: {str(e)}")
                            logger.error(f"Unexpected QR code processing error: {e}")

        conn.close()
    except sqlite3.Error as e:
        st.error(f"خطأ في الوصول إلى قاعدة البيانات: {e}")
        logger.error(f"Database access error: {e}")
# تبويب التقارير
with tabs[7]:
    st.title("التقارير" if st.session_state.language == 'Arabic' else "Reports")
    st.write("عرض وتنزيل تقارير حول حركة المعتمرين في الأدوار." if st.session_state.language == 'Arabic' else "View and download reports on pilgrim movement across floors.")

    # دالة لإنشاء PDF
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            title = 'Crowd Management Report' if st.session_state.language == 'English' else 'تقرير إدارة الحشود'
            self.cell(0, 10, title, 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def generate_pdf_report(data):
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', '', 12)  # Use DejaVu for Arabic support if needed
            for i, row in data.iterrows():
                floor = row['Floor']
                entries = row.get('Entries', 0) if st.session_state.language == 'English' else row.get('دخول', 0)
                exits = row.get('Exits', 0) if st.session_state.language == 'English' else row.get('خروج', 0)
                text = f"{floor}: {entries} entries, {exits} exits" if st.session_state.language == 'English' else f"{floor}: {entries} دخول, {exits} خروج"
                pdf.cell(0, 10, text, 0, 1)
            output = io.BytesIO()
            pdf_data = pdf.output(dest='S').encode('latin1')  # Output to string, encode to bytes
            output.write(pdf_data)
            output.seek(0)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            st.error(f"خطأ في إنشاء تقرير PDF: {e}" if st.session_state.language == 'Arabic' else f"Error generating PDF report: {e}")
            return None

    try:
        conn = sqlite3.connect("crowd_management.db")
        cursor = conn.cursor()

        # التحقق من وجود جدول activity_log
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='activity_log'")
        if not cursor.fetchone():
            st.error("جدول 'activity_log' غير موجود. يرجى إعادة تشغيل التطبيق أو التحقق من قاعدة البيانات.")
            logger.error("Table 'activity_log' does not exist in the database.")
            init_db()  # محاولة إعادة إنشاء الجدول
        else:
            # خيارات التصفية
            st.subheader("تصفية التقارير" if st.session_state.language == 'Arabic' else "Filter Reports")
            period = st.selectbox(
                "اختر الفترة" if st.session_state.language == 'Arabic' else "Select Period",
                ["اليوم", "الأسبوع", "الشهر"] if st.session_state.language == 'Arabic' else ["Today", "This Week", "This Month"],
                key="report_period"
            )
            selected_floor = st.selectbox(
                "اختر الدور (اختياري)" if st.session_state.language == 'Arabic' else "Select Floor (Optional)",
                ["الكل", "الأرضي", "الأول", "الثاني"] if st.session_state.language == 'Arabic' else ["All", "Ground", "First", "Second"],
                key="report_floor"
            )

            # تحديد نطاق الوقت
            end_date = datetime.now()
            if period == "اليوم" or period == "Today":
                start_date = end_date - timedelta(days=1)
            elif period == "الأسبوع" or period == "This Week":
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=30)

            # استرجاع البيانات
            floor_map = {"الأرضي": "Ground", "الأول": "First", "الثاني": "Second"} if st.session_state.language == 'Arabic' else {}
            floor_filter = "" if selected_floor == "الكل" or selected_floor == "All" else f"AND floor = '{floor_map.get(selected_floor, selected_floor)}'"
            query = f"""
                SELECT floor, action, COUNT(*) as count
                FROM activity_log
                WHERE timestamp >= ? AND timestamp <= ? {floor_filter}
                GROUP BY floor, action
            """
            cursor.execute(query, (start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S")))
            report_data = pd.DataFrame(cursor.fetchall(), columns=["Floor", "Action", "Count"])

            # التحقق من وجود بيانات
            if report_data.empty:
                st.warning("لا توجد بيانات متاحة للفترة المحددة. حاول تسجيل بعض الأنشطة في تبويب 'إدارة الأدوار'." if st.session_state.language == 'Arabic' else "No data available for the selected period. Try recording some activities in the 'Floor Management' tab.")
                report_pivot = pd.DataFrame(columns=["Floor", "Entries", "Exits"])
                if st.session_state.language == 'Arabic':
                    report_pivot.columns = ["Floor", "دخول", "خروج"]
            else:
                # إنشاء جدول دمج
                report_pivot = report_data.pivot_table(index="Floor", columns="Action", values="Count", fill_value=0).reset_index()
                logger.info(f"Pivot table columns: {report_pivot.columns.tolist()}")

                # التحقق من الأعمدة الناتجة
                if len(report_pivot.columns) == 1:
                    # إذا كان هناك عمود واحد فقط (Floor)، أضف أعمدة فارغة
                    report_pivot["Entries"] = 0
                    report_pivot["Exits"] = 0
                    if st.session_state.language == 'Arabic':
                        report_pivot.columns = ["Floor", "دخول", "خروج"]
                    else:
                        report_pivot.columns = ["Floor", "Entries", "Exits"]
                else:
                    # إعادة تسمية الأعمدة بناءً على اللغة
                    if "Entry" in report_pivot.columns:
                        report_pivot.columns = ["Floor", "Entries", "Exits"]
                    else:
                        report_pivot.columns = ["Floor", "دخول", "خروج"]

                # تحويل أسماء الأدوار إلى العربية إذا لزم الأمر
                if st.session_state.language == 'Arabic':
                    report_pivot["Floor"] = report_pivot["Floor"].replace({"Ground": "الأرضي", "First": "الأول", "Second": "الثاني"})

            # عرض التقرير
            st.subheader("التقرير" if st.session_state.language == 'Arabic' else "Report")
            st.dataframe(report_pivot, use_container_width=True)

            # رسم بياني
            if not report_pivot.empty:
                fig = px.bar(
                    report_pivot.melt(id_vars="Floor", value_vars=["Entries", "Exits"] if st.session_state.language == 'English' else ["دخول", "خروج"], var_name="Action", value_name="Count"),
                    x="Floor",
                    y="Count",
                    color="Action",
                    barmode="group",
                    title="حركة الدخول والخروج حسب الدور" if st.session_state.language == 'Arabic' else "Entry and Exit Movement by Floor",
                    labels={"Count": "العدد" if st.session_state.language == 'Arabic' else "Count", "Action": "الإجراء" if st.session_state.language == 'Arabic' else "Action"}
                )
                fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("لا توجد بيانات لعرض الرسم البياني." if st.session_state.language == 'Arabic' else "No data available to display the chart.")

            # تنزيل التقرير
            st.subheader("تنزيل التقرير" if st.session_state.language == 'Arabic' else "Download Report")
            csv = report_pivot.to_csv(index=False)
            st.download_button(
                "تنزيل كـ CSV" if st.session_state.language == 'Arabic' else "Download as CSV",
                csv,
                "report.csv",
                "text/csv",
                key="download-csv"
            )
            pdf_data = generate_pdf_report(report_pivot)
            if pdf_data:
                st.download_button(
                    "تنزيل كـ PDF" if st.session_state.language == 'Arabic' else "Download as PDF",
                    pdf_data,
                    "report.pdf",
                    "application/pdf",
                    key="download-pdf"
                )
            else:
                st.error("فشل إنشاء ملف PDF. يرجى التحقق من السجلات لمزيد من التفاصيل." if st.session_state.language == 'Arabic' else "Failed to generate PDF file. Please check the logs for more details.")

        conn.close()
    except sqlite3.Error as e:
        st.error(f"خطأ في الوصول إلى قاعدة البيانات: {e}" if st.session_state.language == 'Arabic' else f"Database access error: {e}")
        logger.error(f"Database access error in reports tab: {e}")
# ------------------------------

# Footer
st.markdown("---")
st.write("تم تطويره لإدارة الحشود في مكة © 2025 | Developed for Makkah Crowd Management © 2025")
# col1, col2 = st.columns([2, 1])
# col1.image("SDA_logo.png",width=250)
# col2.image("lewagon_logo.png" , width=250)
