#PDR-APP中文.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 设置页面配置
st.set_page_config(
    page_title="PDR风险预测系统",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 核心定义：20个特征 ===
# 中英文特征映射字典 - 使用下划线格式
FEATURE_MAPPING = {
    # 基本信息
    '性别': 'Sex',
    '年龄': 'Age',
    '糖尿病病程': 'Course',
    'BMI': 'BMI',
    '腰臀比': 'WHR',
    '收缩压': 'SBP',
    '舒张压': 'DBP',
    '高血压病程': 'duration_of_HT',  # 使用下划线格式
    # 实验室指标
    '血尿素氮': 'BUN',
    '血清肌酐': 'Scr',
    '尿酸': 'UA',
    '总蛋白': 'TP',
    '白蛋白': 'ALB',
    '总胆红素': 'TBIL',
    '直接胆红素': 'DBIL',
    '谷丙转氨酶': 'ALT',
    '谷草转氨酶': 'AST',
    '空腹血糖': 'FBG',
    '糖化血红蛋白': 'HbA1c',
    '尿白蛋白排泄率': 'UAER'
}

# 反向映射（英文到中文）
REVERSE_FEATURE_MAPPING = {v: k for k, v in FEATURE_MAPPING.items()}

# 特征分组
FEATURE_GROUPS = {
    'basic': {
        'name': '基本信息',
        'features': ['性别', '年龄', '糖尿病病程', 'BMI', '腰臀比', '收缩压', '舒张压',
                     '高血压病程']
    },
    'advanced': {
        'name': '实验室指标',
        'features': ['血尿素氮', '血清肌酐', '尿酸', '总蛋白', '白蛋白', '总胆红素', '直接胆红素',
                     '谷丙转氨酶', '谷草转氨酶', '空腹血糖', '糖化血红蛋白', '尿白蛋白排泄率']
    }
}

# 模型需要的特征（20个特征）- 使用下划线格式
MODEL_FEATURES_EN = [
    'Sex', 'Age', 'Course', 'BMI', 'WHR', 'SBP', 'DBP', 'BUN', 'Scr', 'UA',
    'TP', 'ALB', 'TBIL', 'DBIL', 'ALT', 'AST', 'FBG', 'HbA1c', 'UAER', 'duration_of_HT'
]

# 数值型特征 - 使用下划线格式
NUMERIC_FEATURES = [
    'Age', 'Course', 'BMI', 'WHR', 'SBP', 'DBP', 'BUN', 'Scr', 'UA', 'TP',
    'ALB', 'TBIL', 'DBIL', 'ALT', 'AST', 'FBG', 'HbA1c', 'UAER', 'duration_of_HT'
]
CATEGORICAL_FEATURES = ['Sex']

# 分类特征的映射关系
CATEGORY_MAPPINGS = {
    'Sex': {
        'male': 1, '男': 1, '男性': 1, '1': 1, '1.0': 1,
        'female': 2, '女': 2, '女性': 2, '2': 2, '2.0': 2,
        'default': 2  # 默认女性
    }
}

# === 全局变量用于缓存模型 ===
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False


# === 输入字段创建函数 ===
def create_input_field_chinese(feature_name):
    """根据中文特征名创建输入字段"""
    if feature_name == '性别':
        options = [("女性", 2), ("男性", 1)]
        selected = st.selectbox("性别", options=options, format_func=lambda x: x[0])
        return selected[1]
    elif feature_name == '年龄':
        return st.number_input("年龄（岁）", min_value=0, max_value=120, value=50, step=1)
    elif feature_name == '糖尿病病程':
        return st.number_input("糖尿病病程（年）", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    elif feature_name == 'BMI':
        return st.number_input("BMI（体重指数）", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    elif feature_name == '腰臀比':
        return st.number_input("腰臀比（WHR）", min_value=0.5, max_value=1.5, value=0.9, step=0.01)
    elif feature_name == '高血压病程':
        return st.number_input("高血压病程（年）", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    elif feature_name == '收缩压':
        return st.number_input("收缩压（mmHg）", min_value=60.0, max_value=250.0, value=120.0, step=1.0)
    elif feature_name == '舒张压':
        return st.number_input("舒张压（mmHg）", min_value=20.0, max_value=250.0, value=80.0, step=1.0)
    elif feature_name == '血尿素氮':
        return st.number_input("血尿素氮（BUN, mmol/L）", min_value=1.0, max_value=30.0, value=5.0, step=0.1)
    elif feature_name == '血清肌酐':
        return st.number_input("血清肌酐（Scr, μmol/L）", min_value=20.0, max_value=500.0, value=70.0, step=1.0)
    elif feature_name == '尿酸':
        return st.number_input("尿酸（UA, μmol/L）", min_value=100.0, max_value=800.0, value=300.0, step=1.0)
    elif feature_name == '总蛋白':
        return st.number_input("总蛋白（TP, g/L）", min_value=40.0, max_value=100.0, value=70.0, step=0.1)
    elif feature_name == '白蛋白':
        return st.number_input("白蛋白（ALB, g/L）", min_value=20.0, max_value=60.0, value=45.0, step=0.1)
    elif feature_name == '总胆红素':
        return st.number_input("总胆红素（TBIL, μmol/L）", min_value=1.0, max_value=100.0, value=12.0, step=0.1)
    elif feature_name == '直接胆红素':
        return st.number_input("直接胆红素（DBIL, μmol/L）", min_value=0.0, max_value=50.0, value=4.0, step=0.1)
    elif feature_name == '谷丙转氨酶':
        return st.number_input("谷丙转氨酶（ALT, U/L）", min_value=5.0, max_value=200.0, value=25.0, step=1.0)
    elif feature_name == '谷草转氨酶':
        return st.number_input("谷草转氨酶（AST, U/L）", min_value=5.0, max_value=200.0, value=26.0, step=1.0)
    elif feature_name == '空腹血糖':
        return st.number_input("空腹血糖（FBG, mmol/L）", min_value=3.0, max_value=30.0, value=6.5, step=0.1)
    elif feature_name == '糖化血红蛋白':
        return st.number_input("糖化血红蛋白（HbA1c, %）", min_value=4.0, max_value=15.0, value=6.5, step=0.1)
    elif feature_name == '尿白蛋白排泄率':
        return st.number_input("尿白蛋白排泄率（UAER, μg/min）", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
    else:
        return st.number_input(f"{feature_name}", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)


# === 加强版数据清洗函数 ===
def clean_numeric_dataframe(df):
    """清洗数值数据，处理中文逗号/空格/空字符串"""
    df_clean = df.copy()

    for col in df_clean.columns:
        # 先将整个列转换为字符串，然后进行清洗
        df_clean[col] = df_clean[col].astype(str).apply(lambda x: x.strip() if isinstance(x, str) else x)

        def clean_value(x):
            # 如果是NaN或None
            if pd.isna(x):
                return np.nan

            # 如果是字符串
            if isinstance(x, str):
                x_str = x.strip()

                # 如果清洗后是空字符串
                if x_str == '':
                    return np.nan

                # 处理常见的中文符号
                x_str = x_str.replace('，', '.').replace(',', '.').replace(' ', '')

                # 尝试转换为浮点数
                try:
                    return float(x_str)
                except:
                    # 如果转换失败，尝试提取数字部分
                    try:
                        numbers = re.findall(r'-?\d+\.?\d*', x_str)
                        if numbers:
                            return float(numbers[0])
                        else:
                            return np.nan
                    except:
                        return np.nan
            # 如果不是字符串，直接返回
            return x

        df_clean[col] = df_clean[col].apply(clean_value)

    return df_clean


# === 下载链接函数 ===
def get_table_download_link(df, filename="预测结果.xlsx", format="excel"):
    """生成下载链接"""
    df_display = df.copy()
    # 列名转换为中文
    column_mapping = {}
    for col in df_display.columns:
        if col in REVERSE_FEATURE_MAPPING:
            column_mapping[col] = REVERSE_FEATURE_MAPPING[col]
        elif col in ['PDR预测概率', 'PDR预测类别', 'PDR风险等级', '预测时间']:
            column_mapping[col] = col

    if column_mapping:
        df_display = df_display.rename(columns=column_mapping)

    if format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_display.to_excel(writer, index=False, sheet_name='预测结果')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 下载预测结果 (Excel)</a>'
    else:
        csv = df_display.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename.replace(".xlsx", ".csv")}">📥 下载预测结果 (CSV)</a>'
    return href


# === 模型加载函数 ===
@st.cache_resource
def load_model_and_preprocessors():
    """加载模型和预处理工具"""
    try:
        # 加载模型
        model = joblib.load('final_results/lightgbm_pdr_model.pkl')

        # 检查模型的特征名称
        if hasattr(model, 'feature_name_'):
            model_features = model.feature_name_
            st.info(f"📋 模型使用的特征: {model_features}")

        # 加载scaler
        scaler = joblib.load('final_results/scaler.pkl')

        # 检查scaler的特征
        if hasattr(scaler, 'feature_names_in_'):
            st.info(f"📋 标准化器训练时的特征: {scaler.feature_names_in_}")
            st.info(f"📋 标准化器训练时的特征数量: {len(scaler.feature_names_in_)}")

        # 加载imputers
        median_imputer = joblib.load('final_results/median_imputer.pkl')
        mode_imputer = joblib.load('final_results/mode_imputer.pkl')

        # 尝试加载selected_features
        try:
            selected_features = pd.read_csv('final_results/selected_features.csv').iloc[:, 0].tolist()
            # 确保特征顺序与MODEL_FEATURES_EN一致
            selected_features = [f for f in MODEL_FEATURES_EN if f in selected_features]
        except:
            selected_features = MODEL_FEATURES_EN.copy()

        # 特征信息
        feature_info = {
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'selected_features': selected_features,
            'median_imputer': median_imputer,
            'mode_imputer': mode_imputer,
            'scaler': scaler,
            'model_features': MODEL_FEATURES_EN
        }

        return model, scaler, feature_info, selected_features

    except Exception as e:
        st.error(f"❌ 加载失败: {e}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        return None, None, None, None


# === 分类特征标准化函数 ===
def standardize_categorical_feature(series, feature_name):
    """标准化分类特征"""
    if feature_name not in CATEGORY_MAPPINGS:
        return series

    mapping = CATEGORY_MAPPINGS[feature_name]

    def map_value(x):
        if pd.isna(x):
            return mapping['default']

        # 转换为字符串处理
        x_str = str(x).strip().lower()

        # 检查映射
        for key, value in mapping.items():
            if key == 'default':
                continue
            if x_str == key.lower():
                return value

        # 尝试数值转换
        try:
            val = float(x_str)
            if val in [0, 1, 2]:
                return int(val)
        except:
            pass

        # 返回默认值
        return mapping['default']

    return series.apply(map_value)


# === 预处理函数 ===
def preprocess_batch_data(batch_df, feature_info):
    """批量数据预处理"""
    # 1. 数据清洗
    batch_df = clean_numeric_dataframe(batch_df)

    # 2. 列名转换（中文→英文）
    column_mapping = {}
    for col in batch_df.columns:
        col_clean = str(col).strip()
        if col_clean in FEATURE_MAPPING:
            column_mapping[col] = FEATURE_MAPPING[col_clean]
        elif col_clean in MODEL_FEATURES_EN:
            column_mapping[col] = col_clean

    if column_mapping:
        batch_df = batch_df.rename(columns=column_mapping)

    # 3. 强制对齐模型需要的特征
    batch_df_aligned = pd.DataFrame(index=batch_df.index)

    for feature in MODEL_FEATURES_EN:
        if feature in batch_df.columns:
            batch_df_aligned[feature] = batch_df[feature]
        else:
            # 设置默认值
            if feature == 'Sex':
                batch_df_aligned[feature] = 2  # 默认女性
            elif feature == 'Age':
                batch_df_aligned[feature] = 50
            elif feature == 'BMI':
                batch_df_aligned[feature] = 24.0
            elif feature in CATEGORICAL_FEATURES:
                batch_df_aligned[feature] = CATEGORY_MAPPINGS[feature]['default']
            else:
                batch_df_aligned[feature] = 0.0

    # 4. 标准化分类特征
    for cat_feat in CATEGORICAL_FEATURES:
        if cat_feat in batch_df_aligned.columns:
            batch_df_aligned[cat_feat] = standardize_categorical_feature(
                batch_df_aligned[cat_feat], cat_feat
            )

    # 5. 分离数值型/分类型特征
    numeric_data = batch_df_aligned[NUMERIC_FEATURES].copy()
    categorical_data = batch_df_aligned[CATEGORICAL_FEATURES].copy()

    # 6. 确保数值数据是数值类型
    for col in numeric_data.columns:
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')

    # 7. 确保分类数据是整数类型
    for col in categorical_data.columns:
        categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce').fillna(
            CATEGORY_MAPPINGS[col]['default']
        ).astype(int)

    # 8. 填充缺失值
    median_imputer = feature_info.get('median_imputer')
    mode_imputer = feature_info.get('mode_imputer')

    if median_imputer and hasattr(median_imputer, 'feature_names_in_'):
        numeric_filled = median_imputer.transform(numeric_data)
        numeric_data = pd.DataFrame(numeric_filled,
                                    columns=numeric_data.columns,
                                    index=numeric_data.index)
    else:
        # 简单中位数填充
        for col in numeric_data.columns:
            if numeric_data[col].isnull().any():
                median_val = numeric_data[col].median()
                numeric_data[col] = numeric_data[col].fillna(median_val)

    if mode_imputer and hasattr(mode_imputer, 'feature_names_in_'):
        categorical_filled = mode_imputer.transform(categorical_data)
        categorical_data = pd.DataFrame(categorical_filled,
                                        columns=categorical_data.columns,
                                        index=categorical_data.index)
    else:
        # 简单众数填充
        for col in categorical_data.columns:
            if categorical_data[col].isnull().any():
                mode_val = categorical_data[col].mode()[0] if not categorical_data[col].mode().empty else \
                    CATEGORY_MAPPINGS[col]['default']
                categorical_data[col] = categorical_data[col].fillna(mode_val)

    # 9. 合并数据
    processed_data = pd.concat([numeric_data, categorical_data], axis=1)

    # 10. 标准化特征 - 修复版本
    scaler = feature_info.get('scaler')
    if scaler:
        try:
            # 检查标准化器训练时的特征
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
                st.info(f"🔍 标准化器期望的特征: {scaler_features}")

                # 确保我们拥有标准化器需要的所有特征
                missing_features = set(scaler_features) - set(processed_data.columns)
                if missing_features:
                    st.warning(f"⚠️ 标准化器需要以下特征，但数据中缺失: {missing_features}")
                    # 为缺失的特征添加默认值
                    for feat in missing_features:
                        if feat == 'Sex':
                            processed_data[feat] = 2  # 默认女性
                        elif feat in NUMERIC_FEATURES:
                            processed_data[feat] = 0.0
                        else:
                            processed_data[feat] = 0.0

                # 按照标准化器训练时的特征顺序排列数据
                data_for_scaler = processed_data[scaler_features]
                scaled_data = scaler.transform(data_for_scaler)
                # 将标准化后的值放回processed_data
                processed_data[scaler_features] = scaled_data
            else:
                # 如果没有特征名称属性，使用原始方法
                numeric_scaled = scaler.transform(numeric_data)
                processed_data[NUMERIC_FEATURES] = numeric_scaled

        except Exception as e:
            st.warning(f"⚠️ 标准化失败: {e}")
            # 如果失败，使用新的标准化器
            scaler_new = StandardScaler()
            scaled_data = scaler_new.fit_transform(processed_data)
            processed_data = pd.DataFrame(scaled_data, columns=processed_data.columns, index=processed_data.index)

    # 11. 确保特征顺序与MODEL_FEATURES_EN完全一致
    processed_data = processed_data.reindex(columns=MODEL_FEATURES_EN)

    return processed_data, batch_df_aligned


# === 单患者预测函数 ===
def predict_single_patient(input_data_dict, model, feature_info):
    """单患者预测"""
    try:
        # 1. 将输入字典转换为DataFrame
        input_df = pd.DataFrame([input_data_dict])

        # 2. 直接使用批量预处理函数
        processed_data, original_df = preprocess_batch_data(input_df, feature_info)

        if processed_data is None or processed_data.empty:
            st.error("❌ 数据预处理失败")
            return None, None, None

        # 3. 显示调试信息（可选）
        if st.session_state.debug_mode:
            st.write("🔍 调试信息:")
            st.write("处理后数据形状:", processed_data.shape)
            st.write("处理后数据前3行:", processed_data.head(3))

        # 4. 预测
        probability = model.predict_proba(processed_data)[0][1]
        prediction = model.predict(processed_data)[0]

        return probability, prediction, original_df

    except Exception as e:
        st.error(f"预测出错: {e}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None, None, None


# === 批量预测函数 ===
def batch_predict(batch_df, model, feature_info):
    """批量预测"""
    # 预处理
    processed_data, original_df = preprocess_batch_data(batch_df, feature_info)

    if processed_data is None or processed_data.empty:
        st.error("❌ 数据预处理失败")
        return None

    # 显示处理后的数据信息
    st.info(f"📊 处理后数据形状: {processed_data.shape}")

    # 预测
    try:
        probabilities = model.predict_proba(processed_data)[:, 1]
        predictions = model.predict(processed_data)

        # 风险等级
        def get_risk_level(prob):
            if prob < 0.9:
                return "低风险"
            elif prob < 0.99:
                return "中风险"
            else:
                return "高风险"

        risk_levels = [get_risk_level(prob) for prob in probabilities]

        # 结果整合
        results_df = original_df.copy()
        results_df['PDR预测概率'] = probabilities
        results_df['PDR预测类别'] = predictions
        results_df['PDR风险等级'] = risk_levels
        results_df['预测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return results_df

    except Exception as e:
        st.error(f"预测出错: {e}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None


# === 风险等级划分函数 ===
def get_risk_display_info(probability):
    """根据概率获取风险显示信息"""
    if probability < 0.9:
        return "低风险", "✅", "#00D4AA"
    elif probability < 0.99:
        return "中风险", "⚠️", "#FFA500"
    else:
        return "高风险", "🔴", "#FF4B4B"


# === 主函数 ===
def main():
    st.title("👁️ 糖尿病视网膜病变（PDR）风险预测系统")
    st.markdown("---")

    # 加载模型
    with st.spinner("加载模型中..."):
        model, scaler, feature_info, selected_features = load_model_and_preprocessors()

    if model is None:
        st.error("模型加载失败，请检查final_results目录下的模型文件！")
        return

    # 显示模型信息
    if selected_features:
        st.sidebar.header("📊 模型特征信息")
        st.sidebar.info(f"模型使用 {len(selected_features)} 个特征")
        with st.sidebar.expander("查看特征列表"):
            for i, feat in enumerate(MODEL_FEATURES_EN, 1):
                chinese_name = REVERSE_FEATURE_MAPPING.get(feat, feat)
                st.sidebar.write(f"{i}. {chinese_name} ({feat})")

    # 侧边栏设置
    st.sidebar.header("🏥 医疗机构层次")
    facility_level = st.sidebar.radio(
        "选择层次：",
        ["初级（仅基本信息）", "高级（全部指标）"],
        index=1
    )
    selected_groups = ['basic'] if facility_level == "初级（仅基本信息）" else ['basic', 'advanced']

    st.sidebar.header("🔍 预测模式")
    prediction_mode = st.sidebar.radio(
        "选择模式：",
        ["单患者预测", "批量预测"],
        index=0
    )

    # 文件格式设置
    if prediction_mode == "批量预测":
        st.sidebar.header("📁 批量预测设置")
        file_format = st.sidebar.radio("文件格式：", ["Excel (.xlsx)", "CSV (.csv)"], index=0)

        # 生成模板
        st.sidebar.markdown("### 📋 数据模板")
        template_data = {}
        for group_key in selected_groups:
            for feat in FEATURE_GROUPS[group_key]['features']:
                if feat == '性别':
                    template_data[feat] = [1]  # 默认男性
                elif feat == '年龄':
                    template_data[feat] = [50]
                elif feat == '糖尿病病程':
                    template_data[feat] = [5.0]
                elif feat == 'BMI':
                    template_data[feat] = [24.0]
                elif feat == '腰臀比':
                    template_data[feat] = [0.9]
                elif feat == '高血压病程':
                    template_data[feat] = [0.0]
                elif feat == '收缩压':
                    template_data[feat] = [120.0]
                elif feat == '舒张压':
                    template_data[feat] = [80.0]
                elif feat == '血尿素氮':
                    template_data[feat] = [5.0]
                elif feat == '血清肌酐':
                    template_data[feat] = [70.0]
                elif feat == '尿酸':
                    template_data[feat] = [300.0]
                elif feat == '总蛋白':
                    template_data[feat] = [70.0]
                elif feat == '白蛋白':
                    template_data[feat] = [45.0]
                elif feat == '总胆红素':
                    template_data[feat] = [12.0]
                elif feat == '直接胆红素':
                    template_data[feat] = [4.0]
                elif feat == '谷丙转氨酶':
                    template_data[feat] = [25.0]
                elif feat == '谷草转氨酶':
                    template_data[feat] = [26.0]
                elif feat == '空腹血糖':
                    template_data[feat] = [6.5]
                elif feat == '糖化血红蛋白':
                    template_data[feat] = [6.5]
                elif feat == '尿白蛋白排泄率':
                    template_data[feat] = [20.0]
                else:
                    template_data[feat] = [0.0]

        template_df = pd.DataFrame(template_data)

        # 模板下载链接
        if file_format == "Excel (.xlsx)":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='模板')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="pdr预测模板.xlsx">📥 下载Excel模板</a>'
        else:
            csv = template_df.to_csv(index=False, encoding='utf-8-sig')
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="pdr预测模板.csv">📥 下载CSV模板</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

        st.sidebar.markdown("### 📝 使用说明")
        st.sidebar.info("""
        **重要提示：**
        1. 下载模板并填写患者数据
        2. **确保分类列填写正确：**
           - 性别: 1(男) 或 2(女)
        3. **数值列只填写数字**
        4. 上传填写好的文件
        5. 点击"开始批量预测"

        **特征顺序：**
        系统将按照以下顺序处理特征：
        1. 性别 2. 年龄 3. 糖尿病病程 4. BMI 5. 腰臀比 
        6. 收缩压 7. 舒张压 8. 血尿素氮 9. 血清肌酐 
        10. 尿酸 11. 总蛋白 12. 白蛋白 13. 总胆红素 
        14. 直接胆红素 15. 谷丙转氨酶 16. 谷草转氨酶 
        17. 空腹血糖 18. 糖化血红蛋白 19. 尿白蛋白排泄率 
        20. 高血压病程
        """)

    # 添加调试开关
    st.session_state.debug_mode = st.sidebar.checkbox("🔍 启用详细调试模式", value=False)

    st.sidebar.header("ℹ️ 关于")
    st.sidebar.info("本系统基于LightGBM模型，仅供医疗参考！")

    # 单患者预测界面
    if prediction_mode == "单患者预测":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("📝 患者信息输入")
            st.markdown(f"**当前层次：{facility_level}**")

            with st.form("prediction_form"):
                input_values = {}
                # 显示输入字段
                for group_key in selected_groups:
                    group = FEATURE_GROUPS[group_key]
                    st.subheader(group['name'])
                    features = group['features']

                    # 每行显示3个特征
                    num_rows = (len(features) + 2) // 3
                    for row in range(num_rows):
                        row_feats = features[row * 3:(row + 1) * 3]
                        if row_feats:
                            cols = st.columns(3)
                            for idx, feat in enumerate(row_feats):
                                with cols[idx]:
                                    input_values[feat] = create_input_field_chinese(feat)

                # 提交按钮
                submitted = st.form_submit_button("🔍 开始预测", use_container_width=True)

        with col2:
            st.header("📊 预测结果")
            if submitted:
                with st.spinner("分析数据中..."):
                    prob, pred, original_df = predict_single_patient(input_values, model, feature_info)

                    if prob is not None:
                        # 获取风险信息
                        risk_level, risk_icon, risk_color = get_risk_display_info(prob)

                        # 显示原始输入值（验证）
                        if st.session_state.debug_mode:
                            st.subheader("📋 输入验证")
                            st.write("您输入的值:")
                            for i, (feat, val) in enumerate(input_values.items()):
                                st.write(f"{i + 1}. {feat}: {val}")


                        # 结果展示
                        st.metric(f"{risk_icon} 预测结果", risk_level, delta=f"{prob * 100:.2f}%")


                        # 风险解释
                        if risk_level == "高风险":
                            st.error("""
                            ⚠️ **高风险预警**:
                            • 立即进行眼科详细检查
                            • 严格控制血糖/血压
                            • 定期眼底检查
                            • 遵医嘱干预
                            """)
                        elif risk_level == "中风险":
                            st.warning("""
                            ⚠️ **中风险提示**:
                            • 建议进行眼科检查
                            • 加强血糖/血压控制
                            • 每半年复查一次
                            • 注意视力变化
                            """)
                        else:
                            st.success("""
                            ✅ **低风险提示**:
                            • 保持血糖控制
                            • 每年一次眼科检查
                            • 健康生活方式
                            • 视力变化及时就医
                            """)

                        # 提供详细结果下载
                        if original_df is not None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            results_df = original_df.copy()
                            results_df['PDR预测概率'] = prob
                            results_df['PDR预测类别'] = pred
                            results_df['PDR风险等级'] = risk_level
                            results_df['预测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            st.download_button(
                                label="📥 下载详细报告",
                                data=results_df.to_csv(index=False, encoding='utf-8-sig'),
                                file_name=f"单患者预测报告_{timestamp}.csv",
                                mime="text/csv"
                            )

                        st.info("⚠️ 免责声明：结果仅供参考，不替代专业诊断！")
            else:
                st.info("请填写左侧信息并点击预测按钮")

    # 批量预测界面
    else:
        st.header("📁 批量预测")
        st.markdown(f"**当前级别: {facility_level}**")

        # 文件上传
        if file_format == "Excel (.xlsx)":
            uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx', 'xls'],
                                             help="使用左侧模板，支持中文列名！")
        else:
            uploaded_file = st.file_uploader("上传CSV文件", type=['csv'],
                                             help="使用左侧模板，支持中文列名！")

        if uploaded_file is not None:
            try:
                # 读取文件
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    batch_df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    batch_df = pd.read_csv(uploaded_file)

                st.success(f"✅ 成功读取 {len(batch_df)} 行数据！")

                # 数据预览
                st.subheader("📋 数据预览")
                st.write(f"数据形状: {batch_df.shape}")
                st.dataframe(batch_df.head(), use_container_width=True)

                # 批量预测按钮
                if st.button("🚀 开始批量预测", type="primary", use_container_width=True):
                    with st.spinner(f"预测中（共{len(batch_df)}条数据）..."):
                        results_df = batch_predict(batch_df, model, feature_info)

                    if results_df is not None:
                        # 结果展示
                        st.subheader("📊 预测结果")

                        # 统计信息
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总患者数", len(results_df))
                        with col2:
                            high_risk = (results_df['PDR预测概率'] > 0.99).sum()
                            st.metric("高风险", high_risk, delta=f"{high_risk / len(results_df) * 100:.1f}%")
                        with col3:
                            mid_risk = ((results_df['PDR预测概率'] >= 0.9) & (results_df['PDR预测概率'] <= 0.99)).sum()
                            st.metric("中风险", mid_risk, delta=f"{mid_risk / len(results_df) * 100:.1f}%")
                        with col4:
                            low_risk = (results_df['PDR预测概率'] < 0.9).sum()
                            st.metric("低风险", low_risk, delta=f"{low_risk / len(results_df) * 100:.1f}%")

                        # 高风险患者详情
                        high_risk_df = results_df[results_df['PDR风险等级'] == '高风险']
                        if not high_risk_df.empty:
                            st.warning(f"⚠️ 发现 {len(high_risk_df)} 名高风险患者！")
                            with st.expander("🔴 高风险患者详情"):
                                st.dataframe(high_risk_df, use_container_width=True)

                        # 结果展示
                        st.subheader("📋 详细结果")
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            risk_filter = st.selectbox("按风险筛选:", ["全部", "高风险", "中风险", "低风险"])
                        with filter_col2:
                            prob_filter = st.slider("按概率筛选:", 0.0, 1.0, (0.0, 1.0), 0.01)

                        # 应用筛选
                        filtered_df = results_df.copy()
                        if risk_filter != "全部":
                            filtered_df = filtered_df[filtered_df['PDR风险等级'] == risk_filter]
                        filtered_df = filtered_df[
                            (filtered_df['PDR预测概率'] >= prob_filter[0]) &
                            (filtered_df['PDR预测概率'] <= prob_filter[1])
                            ]
                        st.write(f"筛选结果: {len(filtered_df)} 条")
                        st.dataframe(filtered_df, use_container_width=True)

                        # 下载结果
                        st.markdown("### 💾 下载结果")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        col1, col2 = st.columns(2)
                        with col1:
                            excel_name = f"pdr预测结果_{timestamp}.xlsx"
                            st.markdown(get_table_download_link(results_df, excel_name, "excel"),
                                        unsafe_allow_html=True)
                        with col2:
                            csv_name = f"pdr预测结果_{timestamp}.csv"
                            st.markdown(get_table_download_link(results_df, csv_name, "csv"),
                                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"处理文件出错: {str(e)}")
                st.info("""
                **❗ 常见问题解决：**
                1. **检查数据中是否有非数字字符**
                2. **确保分类列填写正确：**
                   - 性别: 1(男) 或 2(女)
                3. **数值列只填写数字**
                4. 下载左侧模板，按模板格式填写数据
                """)
        else:
            # 批量预测说明
            st.info(f"""
            ## 📝 批量预测说明
            1. 下载左侧模板并填写数据
            2. 上传填写好的文件
            3. 点击"开始批量预测"
            4. 查看统计结果并下载

            ### 📋 数据要求
            - 列名与模板一致（中文）
            - **分类列填：**
              - 性别: 1(男) 或 2(女)
            - **数值特征填数字**
            - 系统会自动清洗数据并处理缺失值

            ### 🔢 特征顺序
            系统将按照以下顺序处理特征：
            1. 性别 2. 年龄 3. 糖尿病病程 4. BMI 5. 腰臀比 
            6. 收缩压 7. 舒张压 8. 血尿素氮 9. 血清肌酐 
            10. 尿酸 11. 总蛋白 12. 白蛋白 13. 总胆红素 
            14. 直接胆红素 15. 谷丙转氨酶 16. 谷草转氨酶 
            17. 空腹血糖 18. 糖化血红蛋白 19. 尿白蛋白排泄率 
            20. 高血压病程
            """)

    # 底部信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>基于LightGBM模型 | 仅供医疗专业人员参考</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()