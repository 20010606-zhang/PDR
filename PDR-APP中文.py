import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import sys
import io
import base64
from datetime import datetime
import seaborn as sns
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

# === 核心定义：仅保留20个核心特征 ===
# 中英文特征映射字典
FEATURE_MAPPING = {
    # 基本信息
    '性别': 'Sex',
    '年龄': 'Age',
    '糖尿病病程': 'Course',
    'BMI': 'BMI',
    '腰臀比': 'WHR',
    '收缩压': 'SBP',
    '舒张压': 'DBP',
    '高血压病程': 'duration of HT',
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
        'features': ['性别', '年龄', '糖尿病病程', 'BMI', '腰臀比', '收缩压', '舒张压', '高血压病程']
    },
    'advanced': {
        'name': '实验室指标',
        'features': ['血尿素氮', '血清肌酐', '尿酸', '总蛋白', '白蛋白', '总胆红素', '直接胆红素',
                     '谷丙转氨酶', '谷草转氨酶', '空腹血糖', '糖化血红蛋白', '尿白蛋白排泄率']
    }
}

# 强制定义20个核心特征（英文）
CORE_FEATURES_EN = [
    'Sex', 'Age', 'Course', 'BMI', 'WHR', 'SBP', 'DBP', 'duration of HT',
    'BUN', 'Scr', 'UA', 'TP', 'ALB', 'TBIL', 'DBIL', 'ALT', 'AST', 'FBG', 'HbA1c', 'UAER'
]

# 数值型特征（注意：性别是分类型，其他都是数值型）
NUMERIC_FEATURES = [
    'Age', 'Course', 'BMI', 'WHR', 'SBP', 'DBP', 'duration of HT',
    'BUN', 'Scr', 'UA', 'TP', 'ALB', 'TBIL', 'DBIL', 'ALT', 'AST', 'FBG', 'HbA1c', 'UAER'
]
CATEGORICAL_FEATURES = ['Sex']


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
        return st.number_input("舒张压（mmHg）", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
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

        # 处理空字符串和空白字符
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
                        # 使用正则表达式提取数字（包括小数点和负号）
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


# === 模型加载函数（兼容处理） ===
@st.cache_resource
def load_model_and_preprocessors():
    """加载模型和预处理工具"""
    try:
        # 加载模型
        model = joblib.load('final_results/lightgbm_pdr_model.pkl')

        # 加载scaler
        scaler = joblib.load('final_results/scaler.pkl')

        # 关键：加载建模时使用的imputers
        try:
            median_imputer = joblib.load('feature_selection_results/median_imputer.pkl')
            mode_imputer = joblib.load('feature_selection_results/mode_imputer.pkl')
        except:
            st.warning("⚠️ 未找到建模时的imputer文件，将创建新的imputer")
            median_imputer = SimpleImputer(strategy='median')
            mode_imputer = SimpleImputer(strategy='most_frequent')

        # 尝试加载selected_features，如果没有就使用所有核心特征
        try:
            selected_features = pd.read_csv('final_results/selected_features.csv').iloc[:, 0].tolist()
            # 过滤selected_features，仅保留20个核心特征
            selected_features = [f for f in selected_features if f in CORE_FEATURES_EN]
        except:
            selected_features = CORE_FEATURES_EN

        # 获取imputer拟合时的特征顺序
        try:
            if hasattr(median_imputer, 'feature_names_in_'):
                imputer_numeric_features = list(median_imputer.feature_names_in_)
            else:
                imputer_numeric_features = NUMERIC_FEATURES
        except:
            imputer_numeric_features = NUMERIC_FEATURES

        try:
            if hasattr(mode_imputer, 'feature_names_in_'):
                imputer_categorical_features = list(mode_imputer.feature_names_in_)
            else:
                imputer_categorical_features = CATEGORICAL_FEATURES
        except:
            imputer_categorical_features = CATEGORICAL_FEATURES

        # 特征信息
        feature_info = {
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'feature_names': selected_features,
            'median_imputer': median_imputer,
            'mode_imputer': mode_imputer,
            'scaler': scaler,
            'imputer_numeric_features': imputer_numeric_features,
            'imputer_categorical_features': imputer_categorical_features
        }

        return model, scaler, feature_info, selected_features
    except Exception as e:
        st.error(f"加载失败: {e}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        return None, None, None, None


# === 预处理函数（核心修复） ===
def preprocess_batch_data(batch_df, feature_info, selected_features):
    """批量数据预处理（强制对齐20个核心特征）"""
    # 1. 数据清洗
    batch_df = clean_numeric_dataframe(batch_df)

    # 1.5 确保所有列都是数值类型（除了性别）
    for col in batch_df.columns:
        if col != '性别' and col in batch_df.columns:
            # 尝试将列转换为数值类型
            batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')

    # 2. 列名转换（中文→英文）
    column_mapping = {}
    for col in batch_df.columns:
        col_clean = str(col).strip()
        if col_clean in FEATURE_MAPPING:
            column_mapping[col] = FEATURE_MAPPING[col_clean]
        elif col_clean in CORE_FEATURES_EN:
            column_mapping[col] = col_clean

    if column_mapping:
        batch_df = batch_df.rename(columns=column_mapping)

    # 3. 显示检测到的列
    detected_cols = list(batch_df.columns)
    st.info(f"✅ 检测到 {len(detected_cols)} 个特征列: {detected_cols}")

    # 4. 检查缺失的核心特征
    missing_core = [col for col in CORE_FEATURES_EN if col not in batch_df.columns]
    if missing_core:
        missing_chinese = [REVERSE_FEATURE_MAPPING.get(col, col) for col in missing_core]
        st.warning(f"⚠️ 以下核心特征缺失，将用默认值填充: {missing_chinese}")

    # 5. 强制对齐20个核心特征（只保留需要的，缺失的列填充NaN）
    batch_df_aligned = pd.DataFrame(index=batch_df.index)

    for feature in CORE_FEATURES_EN:
        if feature in batch_df.columns:
            batch_df_aligned[feature] = batch_df[feature]
        else:
            # 设置默认值
            if feature == 'Sex':  # 性别
                batch_df_aligned[feature] = 2  # 默认女性
            elif feature == 'Age':  # 年龄
                batch_df_aligned[feature] = 50  # 默认50岁
            elif feature == 'BMI':  # BMI
                batch_df_aligned[feature] = 24.0  # 默认正常体重
            else:
                batch_df_aligned[feature] = 0.0  # 其他特征默认0

    # 6. 分离数值型/分类型特征
    numeric_data = batch_df_aligned[NUMERIC_FEATURES].copy()
    categorical_data = batch_df_aligned[CATEGORICAL_FEATURES].copy() if CATEGORICAL_FEATURES else pd.DataFrame()

    # 6.5 确保数值数据是数值类型（防止字符串污染）
    for col in numeric_data.columns:
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')

    # 7. 处理性别特征 - 确保是整数类型
    if 'Sex' in categorical_data.columns:
        def convert_sex(x):
            if pd.isna(x):
                return 2

            # 如果是字符串，先处理
            if isinstance(x, str):
                x_str = x.strip()
                if x_str == '':
                    return 2

                # 处理常见性别表示
                if x_str in ['1', '男', 'male', 'Male', 'M', '1.0', '1.00']:
                    return 1
                elif x_str in ['2', '女', 'female', 'Female', 'F', '2.0', '2.00']:
                    return 2
                else:
                    try:
                        val = int(float(x_str))
                        if val == 1:
                            return 1  # 男性
                        else:
                            return 2  # 其他值默认为女性
                    except:
                        return 2  # ✅ 默认女性
            else:
                # 如果不是字符串，直接转换
                try:
                    val = int(float(x))
                    if val == 1:
                        return 1
                    else:
                        return 2
                except:
                    return 2

        categorical_data['Sex'] = categorical_data['Sex'].apply(convert_sex)

    # 8. 填充缺失值 - 使用建模时保存的imputers
    median_imputer = feature_info.get('median_imputer')
    mode_imputer = feature_info.get('mode_imputer')
    imputer_numeric_features = feature_info.get('imputer_numeric_features', NUMERIC_FEATURES)
    imputer_categorical_features = feature_info.get('imputer_categorical_features', CATEGORICAL_FEATURES)

    # 显示数据信息，用于调试
    st.info(f"📊 数值数据形状: {numeric_data.shape}")
    st.info(f"📊 数值数据类型:\n{numeric_data.dtypes}")

    if median_imputer is not None:
        # 确保numeric_data的顺序与imputer拟合时的顺序一致
        numeric_data_reordered = numeric_data.reindex(columns=imputer_numeric_features)

        # 使用建模时的中位数填充器
        try:
            numeric_filled = median_imputer.transform(numeric_data_reordered)

            # 转换回DataFrame，保持原始顺序
            numeric_data = pd.DataFrame(
                numeric_filled,
                columns=imputer_numeric_features,
                index=numeric_data.index
            ).reindex(columns=NUMERIC_FEATURES)

            st.info("✅ 使用建模时的中位数填充数值型特征")
        except Exception as e:
            st.error(f"❌ 中位数填充失败: {e}")
            # 如果失败，使用简单的中位数填充
            st.warning("⚠️ 使用简单中位数填充")
            for col in numeric_data.columns:
                if numeric_data[col].isnull().any():
                    median_val = numeric_data[col].median()
                    numeric_data[col] = numeric_data[col].fillna(median_val)
    else:
        # 如果没有保存的imputer，创建新的
        st.warning("⚠️ 未找到建模时的中位数填充器，将创建新的中位数填充器")
        for col in numeric_data.columns:
            if numeric_data[col].isnull().any():
                median_val = numeric_data[col].median()
                numeric_data[col] = numeric_data[col].fillna(median_val)

    if mode_imputer is not None and not categorical_data.empty:
        # 确保categorical_data的顺序与imputer拟合时的顺序一致
        categorical_data_reordered = categorical_data.reindex(columns=imputer_categorical_features)

        # 使用建模时的众数填充器
        try:
            categorical_filled = mode_imputer.transform(categorical_data_reordered)

            # 转换回DataFrame，保持原始顺序
            categorical_data = pd.DataFrame(
                categorical_filled,
                columns=imputer_categorical_features,
                index=categorical_data.index
            ).reindex(columns=CATEGORICAL_FEATURES)

            st.info("✅ 使用建模时的众数填充分类型特征")
        except Exception as e:
            st.error(f"❌ 众数填充失败: {e}")
            # 如果失败，使用简单的众数填充
            st.warning("⚠️ 使用简单众数填充")
            for col in categorical_data.columns:
                if categorical_data[col].isnull().any():
                    mode_val = categorical_data[col].mode()[0] if not categorical_data[col].mode().empty else 2
                    categorical_data[col] = categorical_data[col].fillna(mode_val)
    elif not categorical_data.empty:
        # 如果没有保存的imputer，创建新的
        st.warning("⚠️ 未找到建模时的众数填充器，将创建新的众数填充器")
        for col in categorical_data.columns:
            if categorical_data[col].isnull().any():
                mode_val = categorical_data[col].mode()[0] if not categorical_data[col].mode().empty else 2
                categorical_data[col] = categorical_data[col].fillna(mode_val)

    # 9. 合并数据
    processed_data = pd.concat([numeric_data, categorical_data], axis=1)

    # 10. 标准化数值型特征 - 使用建模时保存的scaler
    scaler = feature_info.get('scaler')
    if scaler is not None:
        try:
            # 获取scaler拟合时的特征顺序
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
                # 确保numeric_data的顺序与scaler拟合时的顺序一致
                numeric_data_for_scaling = numeric_data.reindex(columns=scaler_features)
                numeric_scaled = scaler.transform(numeric_data_for_scaling)
                numeric_scaled = pd.DataFrame(
                    numeric_scaled,
                    columns=scaler_features,
                    index=numeric_data.index
                ).reindex(columns=NUMERIC_FEATURES)
            else:
                # 如果没有特征名称，直接使用
                numeric_scaled = pd.DataFrame(
                    scaler.transform(numeric_data),
                    columns=NUMERIC_FEATURES,
                    index=numeric_data.index
                )
            st.info("✅ 使用建模时的标准化器进行标准化")
        except Exception as e:
            st.error(f"❌ 标准化失败: {e}")
            st.warning("⚠️ 使用新的标准化器")
            scaler_new = StandardScaler()
            numeric_scaled = pd.DataFrame(
                scaler_new.fit_transform(numeric_data),
                columns=NUMERIC_FEATURES,
                index=numeric_data.index
            )
    else:
        # 如果没有保存的scaler，创建新的
        st.warning("⚠️ 未找到建模时的标准化器，将创建新的标准化器")
        scaler_new = StandardScaler()
        numeric_scaled = pd.DataFrame(
            scaler_new.fit_transform(numeric_data),
            columns=NUMERIC_FEATURES,
            index=numeric_data.index
        )

    processed_data[NUMERIC_FEATURES] = numeric_scaled

    # 11. 确保特征顺序与模型训练时完全一致
    if selected_features:
        # 检查是否有缺失的特征
        missing_features = [f for f in selected_features if f not in processed_data.columns]
        if missing_features:
            st.warning(f"⚠️ 以下模型特征缺失，将用0填充: {missing_features}")
            for feat in missing_features:
                processed_data[feat] = 0

        # 重新排列特征顺序，确保与模型训练时完全一致
        processed_data = processed_data.reindex(columns=selected_features)
        st.info(f"✅ 特征顺序已调整为模型训练时的顺序，共 {len(selected_features)} 个特征")

        # 显示最终的特征顺序
        with st.expander("🔍 查看最终特征顺序"):
            st.write("模型预测时将使用的特征顺序:")
            for i, feat in enumerate(selected_features, 1):
                chinese_name = REVERSE_FEATURE_MAPPING.get(feat, feat)
                st.write(f"{i}. {chinese_name} ({feat})")
    else:
        # 如果没有指定selected_features，使用所有核心特征
        processed_data = processed_data[CORE_FEATURES_EN]

    return processed_data, batch_df_aligned


# === 批量预测函数 ===
def batch_predict(batch_df, model, feature_info, selected_features):
    """批量预测"""
    # 预处理
    processed_data, original_df = preprocess_batch_data(batch_df, feature_info, selected_features)

    if processed_data is None:
        return None

    # 显示处理后的数据形状和特征顺序
    st.info(f"📊 处理后数据形状: {processed_data.shape}")
    st.info(f"📊 使用的特征数量: {len(processed_data.columns)}")

    # 显示前几行数据用于调试
    with st.expander("🔍 预处理后数据预览"):
        st.dataframe(processed_data.head(), use_container_width=True)

    # 预测
    try:
        probabilities = model.predict_proba(processed_data)[:, 1]
        predictions = model.predict(processed_data)

        # 风险等级
        def get_risk_level(prob):
            if prob < 0.3:
                return "低风险"
            elif prob < 0.7:
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


# === 单患者预测函数 ===
def predict_single_patient(input_data_dict, model, feature_info):
    """单患者预测"""
    try:
        # 转换为英文特征名
        english_input = {}
        for chinese_feature, value in input_data_dict.items():
            if chinese_feature in FEATURE_MAPPING:
                english_input[FEATURE_MAPPING[chinese_feature]] = value

        # 填充缺失特征的默认值
        for feature in CORE_FEATURES_EN:
            if feature not in english_input:
                if feature == 'Sex':
                    english_input[feature] = 2
                elif feature == 'Age':
                    english_input[feature] = 50
                elif feature == 'BMI':
                    english_input[feature] = 24.0
                else:
                    english_input[feature] = 0.0

        # 转换为DataFrame
        input_df = pd.DataFrame([english_input], columns=CORE_FEATURES_EN)

        # 预处理
        numeric_data = input_df[NUMERIC_FEATURES].copy()
        categorical_data = input_df[CATEGORICAL_FEATURES].copy()

        # 预处理性别
        if 'Sex' in categorical_data.columns:
            def convert_sex(x):
                if pd.isna(x):
                    return 2
                x_str = str(x).strip()
                if x_str in ['1', '男', 'male', 'Male', 'M']:
                    return 1
                elif x_str in ['2', '女', 'female', 'Female', 'F']:
                    return 2
                else:
                    try:
                        val = int(float(x_str))
                        if val == 1:
                            return 1
                        else:
                            return 2
                    except:
                        return 2  # ✅ 女性

            categorical_data['Sex'] = categorical_data['Sex'].apply(convert_sex)

        # 使用建模时保存的scaler进行标准化
        scaler = feature_info.get('scaler')
        if scaler is not None:
            numeric_scaled = scaler.transform(numeric_data)
            numeric_data = pd.DataFrame(numeric_scaled, columns=NUMERIC_FEATURES)
        else:
            # 如果没有保存的scaler，创建新的
            scaler_new = StandardScaler()
            numeric_scaled = scaler_new.fit_transform(numeric_data)
            numeric_data = pd.DataFrame(numeric_scaled, columns=NUMERIC_FEATURES)

        # 合并数据
        processed_data = pd.concat([numeric_data, categorical_data], axis=1)

        # 使用模型需要的特征并确保顺序一致
        if 'feature_names' in feature_info and feature_info['feature_names']:
            selected_features = feature_info['feature_names']
            # 确保所有特征都存在
            for feat in selected_features:
                if feat not in processed_data.columns:
                    processed_data[feat] = 0
            # 重新排列特征顺序
            processed_data = processed_data[selected_features]
        else:
            processed_data = processed_data[CORE_FEATURES_EN]

        # 预测
        probability = model.predict_proba(processed_data)[0][1]
        prediction = model.predict(processed_data)[0]

        return probability, prediction
    except Exception as e:
        st.error(f"预测出错: {e}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None, None


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

    # 检查imputer是否加载成功
    if feature_info.get('median_imputer') is None or feature_info.get('mode_imputer') is None:
        st.warning("""
        ⚠️ **注意**：未找到建模时的填充器文件
        - 预测时将使用当前数据重新计算中位数/众数
        - 这可能导致与建模时不一致，影响预测准确性
        - 请确保将 `median_imputer.pkl` 和 `mode_imputer.pkl` 放在 `feature_selection_results/` 目录下
        """)

    # 检查scaler是否加载成功
    if feature_info.get('scaler') is None:
        st.warning("""
        ⚠️ **注意**：未找到建模时的标准化器文件
        - 预测时将使用当前数据重新计算标准化参数
        - 这可能导致与建模时不一致，影响预测准确性
        - 请确保将 `scaler.pkl` 放在 `final_results/` 目录下
        """)

    # 显示模型使用的特征
    if selected_features:
        st.sidebar.header("📊 模型特征信息")
        st.sidebar.info(f"模型使用 {len(selected_features)} 个特征")
        with st.sidebar.expander("查看特征列表"):
            for i, feat in enumerate(selected_features, 1):
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
    file_format = "Excel (.xlsx)"
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
                elif feat in ['收缩压', '舒张压']:
                    template_data[feat] = [120.0 if feat == '收缩压' else 80.0]
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
        2. **确保性别列为：1(男) 或 2(女)**
        3. **数值列只填写数字，不要有中文或符号**
        4. **空单元格或缺失数据请留空，不要填写空格**
        5. 上传填写好的文件
        6. 点击"开始批量预测"
        7. 查看并下载结果

        **支持的性别格式：**
        - 数字: 1 (男), 2 (女)
        - 中文: 男, 女
        - 英文: male, female, M, F

        **数据清洗规则：**
        - 系统会自动清除空格、中文逗号等非数字字符
        - 空字符串会被视为缺失值
        - 非数字字符会被提取数字部分或转换为NaN
        """)

    st.sidebar.header("ℹ️ 关于")
    st.sidebar.info("本系统基于LightGBM模型，仅供医疗参考！")
    st.sidebar.header("📊 模型信息")
    st.sidebar.text(f"层次：{facility_level}")
    st.sidebar.text(f"模式：{prediction_mode}")

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
                    # 排版优化
                    if len(features) <= 5:
                        cols = st.columns(len(features))
                        for idx, feat in enumerate(features):
                            with cols[idx]:
                                input_values[feat] = create_input_field_chinese(feat)
                    else:
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
                    prob, pred = predict_single_patient(input_values, model, feature_info)
                    if prob is not None:
                        # 可视化风险概率（单患者预测保留）
                        fig, ax = plt.subplots(figsize=(8, 4))
                        risk_level = "高风险" if pred == 1 else "低风险"
                        color = '#FF4B4B' if pred == 1 else '#00D4AA'
                        ax.barh([0], [prob * 100], color=color, alpha=0.7)
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('PDR风险概率 (%)')
                        ax.set_yticks([])
                        ax.set_title(f'风险概率: {prob * 100:.2f}%')
                        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='阈值')
                        ax.legend()
                        st.pyplot(fig)

                        # 结果展示
                        st.metric("预测结果", risk_level, delta=f"{prob * 100:.2f}%")

                        # 使用的特征
                        with st.expander("📋 使用的特征"):
                            used_feats = []
                            for g in selected_groups:
                                used_feats.extend(FEATURE_GROUPS[g]['features'])
                            st.write(f"总数: {len(used_feats)}")
                            for i, feat in enumerate(used_feats, 1):
                                st.write(f"{i}. {feat}")

                        # 建议
                        st.subheader("💡 建议")
                        if pred == 1:
                            st.error("""
                            ⚠️ **高风险预警**:
                            • 立即进行眼科详细检查
                            • 严格控制血糖/血压
                            • 定期眼底检查
                            • 遵医嘱干预
                            """)
                        else:
                            st.success("""
                            ✅ **低风险提示**:
                            • 保持血糖控制
                            • 每年一次眼科检查
                            • 健康生活方式
                            • 视力变化及时就医
                            """)
                        st.info("⚠️ 免责声明：结果仅供参考，不替代专业诊断！")
            else:
                st.info("请填写左侧信息并点击预测按钮")
                with st.expander("📋 可用特征"):
                    used_feats = []
                    for g in selected_groups:
                        group = FEATURE_GROUPS[g]
                        st.write(f"**{group['name']}** ({len(group['features'])}个):")
                        for i, feat in enumerate(group['features'], 1):
                            st.write(f"  {i}. {feat}")
                        used_feats.extend(group['features'])
                    st.write(f"\n**总特征数: {len(used_feats)}**")

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
                st.write(f"列名: {list(batch_df.columns)}")
                st.dataframe(batch_df.head(), use_container_width=True)

                # 数据统计
                with st.expander("📊 数据统计"):
                    st.write("**数值特征统计:**")
                    numeric_cols = batch_df.select_dtypes(include=[np.number]).columns
                    if numeric_cols.empty:
                        st.write("无数值型特征")
                    else:
                        st.dataframe(batch_df[numeric_cols].describe())

                    st.write("**缺失值统计:**")
                    missing = batch_df.isnull().sum()
                    missing_df = pd.DataFrame({
                        '特征': missing.index,
                        '缺失数': missing.values,
                        '缺失率(%)': (missing / len(batch_df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['缺失数'] > 0]
                    if missing_df.empty:
                        st.success("✅ 无缺失值！")
                    else:
                        st.dataframe(missing_df)

                    st.write("**数据类型统计:**")
                    dtypes_df = pd.DataFrame({
                        '特征': batch_df.columns,
                        '数据类型': batch_df.dtypes.values
                    })
                    st.dataframe(dtypes_df)

                # 批量预测按钮
                if st.button("🚀 开始批量预测", type="primary", use_container_width=True):
                    with st.spinner(f"预测中（共{len(batch_df)}条数据）..."):
                        results_df = batch_predict(batch_df, model, feature_info, selected_features)

                    if results_df is not None:
                        # 结果展示
                        st.subheader("📊 预测结果")
                        # 统计信息
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总患者数", len(results_df))
                        with col2:
                            high_risk = (results_df['PDR预测概率'] > 0.95).sum()
                            st.metric("高风险", high_risk)
                        with col3:
                            mid_risk = ((results_df['PDR预测概率'] >= 0.5) & (results_df['PDR预测概率'] <= 0.95)).sum()
                            st.metric("中风险", mid_risk)
                        with col4:
                            low_risk = (results_df['PDR预测概率'] < 0.5).sum()
                            st.metric("低风险", low_risk)

                        # 高风险患者详情
                        high_risk_df = results_df[results_df['PDR风险等级'] == '高风险']
                        if not high_risk_df.empty:
                            st.warning(f"⚠️ 发现 {len(high_risk_df)} 名高风险患者！")
                            with st.expander("🔴 高风险患者详情"):
                                st.dataframe(high_risk_df, use_container_width=True)

                        # 结果筛选
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

                        # 高风险患者单独下载
                        if not high_risk_df.empty:
                            st.markdown("#### 🔴 高风险患者下载")
                            high_name = f"pdr高风险患者_{timestamp}.xlsx"
                            st.markdown(get_table_download_link(high_risk_df, high_name, "excel"),
                                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"处理文件出错: {str(e)}")
                import traceback
                st.error(f"详细错误信息: {traceback.format_exc()}")
                st.info("""
                **❗ 常见问题解决：**
                1. **检查数据中是否有非数字字符（如空格、中文、特殊符号）**
                2. **确保性别列仅填写1/2，或使用中文"男"/"女"**
                3. **数值列只填写数字，不要有单位或符号**
                4. **下载左侧模板，按模板格式填写数据**
                5. 确保包含所有必需的特征列
                6. **空单元格请留空，不要填写空格或任何字符**
                """)
        else:
            # 批量预测说明
            st.info(f"""
            ## 📝 批量预测说明（{file_format}）
            1. 下载左侧模板并填写数据
            2. 上传填写好的{file_format}文件
            3. 点击"开始批量预测"
            4. 查看统计结果并下载

            ### 📋 数据要求
            - 列名与模板一致（中文）
            - **性别列填：1(男) 或 2(女)**（支持多种格式）
            - **数值特征填数字，不要有单位或符号**
            - **空白单元格或缺失数据请留空，不要填写空格**
            - 系统会自动清洗数据并处理缺失值
            """)

    # 底部信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>基于LightGBM模型 | 仅供医疗专业人员参考</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()