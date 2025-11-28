import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import roc_curve, auc

# 设置页面配置
st.set_page_config(
    page_title="PDR Risk Prediction System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 兼容部署环境
plt.rcParams['axes.unicode_minus'] = False

# 显示环境信息
st.sidebar.header("Environment Info")
st.sidebar.text(f"Python: {sys.version.split()[0]}")
try:
    import lightgbm as lgb

    st.sidebar.text(f"LightGBM: {lgb.__version__}")
except ImportError:
    st.sidebar.text("LightGBM: Not installed")


# 加载模型和预处理工具
@st.cache_resource
def load_model_and_preprocessors():
    try:
        # 获取当前文件目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼接模型文件路径
        model_path = os.path.join(current_dir, 'lightgbm_pdr_model.pkl')
        scaler_path = os.path.join(current_dir, 'scaler.pkl')
        median_imputer_path = os.path.join(current_dir, 'median_imputer.pkl')
        mode_imputer_path = os.path.join(current_dir, 'mode_imputer.pkl')
        feature_info_path = os.path.join(current_dir, 'feature_info.pkl')

        # 检查文件是否存在
        missing_files = []
        for file_path, name in [
            (model_path, "Model"),
            (scaler_path, "Scaler"),
            (median_imputer_path, "Median Imputer"),
            (mode_imputer_path, "Mode Imputer"),
            (feature_info_path, "Feature Info")
        ]:
            if not os.path.exists(file_path):
                missing_files.append(name)

        if missing_files:
            st.warning(f"Missing files: {', '.join(missing_files)}")

        model = joblib.load(model_path) if os.path.exists(model_path) else None
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        median_imputer = joblib.load(median_imputer_path) if os.path.exists(median_imputer_path) else None
        mode_imputer = joblib.load(mode_imputer_path) if os.path.exists(mode_imputer_path) else None
        feature_info = pickle.load(open(feature_info_path, 'rb')) if os.path.exists(feature_info_path) else None

        return model, scaler, median_imputer, mode_imputer, feature_info

    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        st.info("Please install required packages: pip install -r requirements.txt")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, None, None


# 预测函数
def predict_pdr_risk(input_data, model, scaler, median_imputer, mode_imputer, feature_info):
    try:
        if feature_info is None:
            st.error("Feature info not loaded")
            return 0.5, 0

        # 转换为DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_info['feature_names'])

        # 数据预处理
        if median_imputer:
            numeric_data = median_imputer.transform(input_df[feature_info['numeric_features']])
        else:
            numeric_data = input_df[feature_info['numeric_features']].values

        if mode_imputer:
            categorical_data = mode_imputer.transform(input_df[feature_info['categorical_features']])
        else:
            categorical_data = input_df[feature_info['categorical_features']].values

        # 重新组合
        processed_data = np.column_stack([numeric_data, categorical_data])
        processed_df = pd.DataFrame(processed_data, columns=feature_info['feature_names'])

        # 标准化
        if scaler:
            scaled_data = scaler.transform(processed_df)
        else:
            scaled_data = processed_data

        # 预测
        if model:
            probability = model.predict_proba(scaled_data)[0][1] if hasattr(model, 'predict_proba') else 0.5
            prediction = model.predict(scaled_data)[0] if hasattr(model, 'predict') else 0
        else:
            probability = 0.5
            prediction = 0

        return probability, prediction

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return 0.5, 0


# 主应用
def main():
    # 标题和介绍
    st.title("👁️ Diabetic Retinopathy (PDR) Risk Prediction System")
    st.markdown("---")

    # 加载模型
    with st.spinner("Loading prediction model..."):
        model, scaler, median_imputer, mode_imputer, feature_info = load_model_and_preprocessors()

    if model is None and feature_info is None:
        st.error("Unable to load model and feature info. Please check if model files exist.")
        return

    # 侧边栏
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This system is based on a LightGBM machine learning model for predicting the risk of Proliferative Diabetic Retinopathy (PDR). "
        "Please enter patient clinical indicators to obtain risk assessment."
    )

    if feature_info:
        st.sidebar.header("📊 Model Information")
        st.sidebar.text(f"Number of features: {len(feature_info['feature_names'])}")
        st.sidebar.text(f"Target variable: {feature_info['target_name']}")

    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Patient Information Input")
        st.markdown("Please fill in the following patient clinical indicators:")

        # 创建表单
        with st.form("prediction_form"):
            # 基本信息
            st.subheader("Basic Information")
            col1_1, col1_2, col1_3 = st.columns(3)

            with col1_1:
                sex = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
                smoking = st.selectbox("Smoking History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[
                    1]

            with col1_2:
                drinking = \
                st.selectbox("Drinking History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
                course = st.number_input("Diabetes Duration (years)", min_value=0.0, max_value=50.0, value=5.0,
                                         step=0.5)
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)

            with col1_3:
                whr = st.number_input("Waist-Hip Ratio (WHR)", min_value=0.5, max_value=1.5, value=0.9, step=0.01)
                ht = st.selectbox("Hypertension History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[
                    1]
                ht_duration = st.number_input("Hypertension Duration (years)", min_value=0.0, max_value=50.0, value=0.0,
                                              step=0.5)

            # 血压和实验室指标
            st.subheader("Blood Pressure and Laboratory Indicators")
            col2_1, col2_2 = st.columns(2)

            with col2_1:
                sbp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60.0, max_value=250.0, value=120.0,
                                      step=1.0)
                dbp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40.0, max_value=150.0, value=80.0,
                                      step=1.0)
                bun = st.number_input("Blood Urea Nitrogen (BUN, mmol/L)", min_value=1.0, max_value=30.0, value=5.0,
                                      step=0.1)
                scr = st.number_input("Serum Creatinine (Scr, μmol/L)", min_value=20.0, max_value=500.0, value=70.0,
                                      step=1.0)
                ua = st.number_input("Uric Acid (UA, μmol/L)", min_value=100.0, max_value=800.0, value=300.0, step=1.0)

            with col2_2:
                tp = st.number_input("Total Protein (TP, g/L)", min_value=40.0, max_value=100.0, value=70.0, step=0.1)
                alb = st.number_input("Albumin (ALB, g/L)", min_value=20.0, max_value=60.0, value=45.0, step=0.1)
                tbil = st.number_input("Total Bilirubin (TBIL, μmol/L)", min_value=1.0, max_value=100.0, value=12.0,
                                       step=0.1)
                dbil = st.number_input("Direct Bilirubin (DBIL, μmol/L)", min_value=0.0, max_value=50.0, value=4.0,
                                       step=0.1)

            # 肝功能和其他指标
            st.subheader("Liver Function and Other Indicators")
            col3_1, col3_2 = st.columns(2)

            with col3_1:
                alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=200.0, value=25.0, step=1.0)
                ast = st.number_input("AST (U/L)", min_value=5.0, max_value=200.0, value=26.0, step=1.0)
                fbg = st.number_input("Fasting Blood Glucose (FBG, mmol/L)", min_value=3.0, max_value=30.0, value=6.5,
                                      step=0.1)

            with col3_2:
                hba1c = st.number_input("Glycated Hemoglobin (HbA1c, %)", min_value=4.0, max_value=15.0, value=6.5,
                                        step=0.1)
                uaer = st.number_input("Urinary Albumin Excretion Rate (UAER, μg/min)", min_value=0.0, max_value=500.0,
                                       value=20.0, step=1.0)

            # 提交按钮
            submitted = st.form_submit_button("🔍 Start Prediction", use_container_width=True)

    # 预测结果显示区域
    with col2:
        st.header("📊 Prediction Results")

        if submitted:
            with st.spinner("Analyzing data..."):
                # 准备输入数据
                input_data = [
                    sex, age, smoking, drinking, course, bmi, whr, sbp, dbp,
                    bun, scr, ua, tp, alb, tbil, dbil, alt, ast, fbg,
                    hba1c, uaer, ht, ht_duration
                ]

                # 进行预测
                probability, prediction = predict_pdr_risk(
                    input_data, model, scaler, median_imputer, mode_imputer, feature_info
                )

                # 显示风险概率
                st.subheader("Risk Assessment")

                # 创建仪表盘
                fig, ax = plt.subplots(figsize=(8, 4))
                risk_level = "High Risk" if prediction == 1 else "Low Risk"
                colors = ['#FF4B4B', '#00D4AA']
                color = colors[1] if prediction == 0 else colors[0]

                ax.barh([0], [probability * 100], color=color, alpha=0.7)
                ax.set_xlim(0, 100)
                ax.set_xlabel('PDR Risk Probability (%)')
                ax.set_yticks([])
                ax.set_title(f'Risk Probability: {probability * 100:.2f}%')

                # 添加风险阈值线
                ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
                ax.legend()

                st.pyplot(fig)

                # 显示详细结果
                st.metric(
                    label="Prediction Result",
                    value=risk_level,
                    delta=f"{probability * 100:.2f}%"
                )

                # 建议信息
                st.subheader("💡 Recommendations")
                if prediction == 1:
                    st.error(
                        "⚠️ **High Risk Alert**:\n\n"
                        "• Recommend immediate detailed ophthalmological examination\n"
                        "• Strictly control blood glucose and blood pressure\n"
                        "• Regular fundus examination\n"
                        "• Follow medical advice for necessary treatment interventions"
                    )
                else:
                    st.success(
                        "✅ **Low Risk Alert**:\n\n"
                        "• Continue maintaining good blood glucose control\n"
                        "• Annual ophthalmological examination\n"
                        "• Maintain healthy lifestyle\n"
                        "• Seek medical attention promptly if vision changes occur"
                    )

                # 免责声明
                st.info(
                    "**Disclaimer**: This prediction result is based on a machine learning model and is for reference only. "
                    "It cannot replace professional medical diagnosis. Please consult healthcare professionals if you have any questions."
                )

        else:
            # 默认显示等待信息
            st.info("Please fill in patient information on the left and click 'Start Prediction' button")

            # 显示特征重要性（可选）
            st.subheader("📈 Important Features")
            st.write("The model primarily considers the following key features:")
            important_features = [
                "Glycated Hemoglobin (HbA1c)", "Diabetes Duration", "Age",
                "Urinary Albumin Excretion Rate (UAER)", "Systolic Blood Pressure (SBP)", "Fasting Blood Glucose (FBG)"
            ]
            for i, feature in enumerate(important_features, 1):
                st.write(f"{i}. {feature}")

    # 底部信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Based on LightGBM Machine Learning Model | For Medical Professionals Reference Only"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()