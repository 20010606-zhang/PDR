import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import io
import base64
from datetime import datetime
import seaborn as sns

# 设置页面配置
st.set_page_config(
    page_title="PDR风险预测系统",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 显示环境信息（调试用）
st.sidebar.header("环境信息")
st.sidebar.text(f"Python: {sys.version.split()[0]}")

# === 核心修改1：定义两级特征组 (原为三级) ===
FEATURE_GROUPS = {
    'basic': {
        'name': '基本信息',
        'features': ['性别', '年龄', '糖尿病病程', 'BMI', '腰臀比','收缩压', '舒张压',
                     '高血压病程']
    },
    'advanced': {
        'name': '血压和实验室指标',
        'features': ['血尿素氮', '血清肌酐', '尿酸', '总蛋白', '白蛋白', '总胆红素', '直接胆红素', '谷丙转氨酶', '谷草转氨酶', '空腹血糖', '糖化血红蛋白', '尿白蛋白排泄率']
    }
}
# === 核心修改1结束 ===

# 创建下载链接的函数
def get_table_download_link(df, filename="预测结果.xlsx", format="excel"):
    """生成Excel或CSV下载链接"""
    if format == "excel":
        # 生成Excel文件
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='预测结果')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 下载预测结果 (Excel)</a>'
    else:
        # 生成CSV文件
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename.replace(".xlsx", ".csv")}">📥 下载预测结果 (CSV)</a>'
    return href

# 加载模型和预处理工具
@st.cache_resource
def load_model_and_preprocessors():
    try:
        model = joblib.load('final_results/lightgbm_pdr_model.pkl')
        scaler = joblib.load('final_results/scaler.pkl')
        median_imputer = joblib.load('final_results/median_imputer.pkl')
        mode_imputer = joblib.load('final_results/mode_imputer.pkl')

        with open('final_results/feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)

        # 加载选择的特征
        selected_features = pd.read_csv('final_results/selected_features.csv').iloc[:, 0].tolist()

        return model, scaler, median_imputer, mode_imputer, feature_info, selected_features
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None, None, None, None, None, None

# 预处理批量数据
def preprocess_batch_data(batch_df, feature_info, median_imputer, mode_imputer, scaler, selected_features):
    """批量预处理数据"""

    # 确保所有需要的列都存在
    required_columns = feature_info['feature_names']
    missing_cols = set(required_columns) - set(batch_df.columns)

    if missing_cols:
        st.warning(f"⚠️ 数据中缺少以下列：{list(missing_cols)}")
        # 为缺失的列填充NaN
        for col in missing_cols:
            batch_df[col] = np.nan

    # 确保列顺序一致
    batch_df = batch_df[required_columns]

    # 复制一份原始数据用于结果输出
    original_df = batch_df.copy()

    # 分别处理数值型和分类型特征
    numeric_data = batch_df[feature_info['numeric_features']].copy()
    categorical_data = batch_df[feature_info['categorical_features']].copy()

    # 处理缺失值
    if numeric_data.isnull().any().any():
        missing_count = numeric_data.isnull().sum().sum()
        st.info(f"🔍 数值型特征缺失值数量：{missing_count}")
        numeric_data = pd.DataFrame(
            median_imputer.transform(numeric_data),
            columns=feature_info['numeric_features']
        )

    if categorical_data.isnull().any().any():
        missing_count = categorical_data.isnull().sum().sum()
        st.info(f"🔍 分类型特征缺失值数量：{missing_count}")
        categorical_data = pd.DataFrame(
            mode_imputer.transform(categorical_data),
            columns=feature_info['categorical_features']
        )

    # 合并数据
    processed_data = pd.concat([numeric_data, categorical_data], axis=1)

    # 标准化数值型特征
    numeric_features_standardized = pd.DataFrame(
        scaler.transform(processed_data[feature_info['numeric_features']]),
        columns=feature_info['numeric_features']
    )

    # 更新数值型特征
    processed_data[feature_info['numeric_features']] = numeric_features_standardized

    # 只保留选择的特征（与训练时相同）
    processed_data_selected = processed_data[selected_features]

    return processed_data_selected, original_df

# 批量预测函数
def batch_predict(batch_df, model, feature_info, median_imputer, mode_imputer, scaler, selected_features):
    """批量预测"""

    # 预处理数据
    processed_data, original_df = preprocess_batch_data(
        batch_df, feature_info, median_imputer, mode_imputer, scaler, selected_features
    )

    # 进行预测
    probabilities = model.predict_proba(processed_data)[:, 1]
    predictions = model.predict(processed_data)

    # 确定风险等级
    def get_risk_level(prob):
        if prob < 0.3:
            return "低风险"
        elif prob < 0.7:
            return "中风险"
        else:
            return "高风险"

    risk_levels = [get_risk_level(prob) for prob in probabilities]

    # 创建结果DataFrame
    results_df = original_df.copy()
    results_df['PDR预测概率'] = probabilities
    results_df['PDR预测类别'] = predictions
    results_df['PDR风险等级'] = risk_levels
    results_df['预测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return results_df

# 创建输入字段函数（单个患者用）
def create_input_field(feature_name, feature_type, feature_info):
    """根据特征类型创建不同的输入字段"""

    if feature_type == 'numeric':
        # 数值型特征
        if feature_name == '性别':
            options = [("女性", 0), ("男性", 1)]
            selected = st.selectbox("性别", options=options, format_func=lambda x: x[0])
            return selected[1]
        elif feature_name == '吸烟史':
            options = [("否", 0), ("是", 1)]
            selected = st.selectbox("吸烟史", options=options, format_func=lambda x: x[0])
            return selected[1]
        elif feature_name == '饮酒史':
            options = [("否", 0), ("是", 1)]
            selected = st.selectbox("饮酒史", options=options, format_func=lambda x: x[0])
            return selected[1]
        elif feature_name == '高血压病史':
            options = [("否", 0), ("是", 1)]
            selected = st.selectbox("高血压病史", options=options, format_func=lambda x: x[0])
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
            return st.number_input("收缩压（mmHg）", min_value=60.0, max_value=250.0, value=120.0,
                                   step=1.0)
        elif feature_name == '舒张压':
            return st.number_input("舒张压（mmHg）", min_value=40.0, max_value=150.0, value=80.0,
                                   step=1.0)
        elif feature_name == '血尿素氮':
            return st.number_input("血尿素氮（BUN, mmol/L）", min_value=1.0, max_value=30.0, value=5.0,
                                   step=0.1)
        elif feature_name == '血清肌酐':
            return st.number_input("血清肌酐（Scr, μmol/L）", min_value=20.0, max_value=500.0, value=70.0,
                                   step=1.0)
        elif feature_name == '尿酸':
            return st.number_input("尿酸（UA, μmol/L）", min_value=100.0, max_value=800.0, value=300.0, step=1.0)
        elif feature_name == '总蛋白':
            return st.number_input("总蛋白（TP, g/L）", min_value=40.0, max_value=100.0, value=70.0, step=0.1)
        elif feature_name == '白蛋白':
            return st.number_input("白蛋白（ALB, g/L）", min_value=20.0, max_value=60.0, value=45.0, step=0.1)
        elif feature_name == '总胆红素':
            return st.number_input("总胆红素（TBIL, μmol/L）", min_value=1.0, max_value=100.0, value=12.0,
                                   step=0.1)
        elif feature_name == '直接胆红素':
            return st.number_input("直接胆红素（DBIL, μmol/L）", min_value=0.0, max_value=50.0, value=4.0,
                                   step=0.1)
        elif feature_name == '谷丙转氨酶':
            return st.number_input("谷丙转氨酶（ALT, U/L）", min_value=5.0, max_value=200.0, value=25.0, step=1.0)
        elif feature_name == '谷草转氨酶':
            return st.number_input("谷草转氨酶（AST, U/L）", min_value=5.0, max_value=200.0, value=26.0, step=1.0)
        elif feature_name == '空腹血糖':
            return st.number_input("空腹血糖（FBG, mmol/L）", min_value=3.0, max_value=30.0, value=6.5,
                                   step=0.1)
        elif feature_name == '糖化血红蛋白':
            return st.number_input("糖化血红蛋白（HbA1c, %）", min_value=4.0, max_value=15.0, value=6.5, step=0.1)
        elif feature_name == '尿白蛋白排泄率':
            return st.number_input("尿白蛋白排泄率（UAER, μg/min）", min_value=0.0, max_value=500.0,
                                   value=20.0, step=1.0)
        else:
            return st.number_input(f"{feature_name}", min_value=0, max_value=100, value=50, step=1.0)

    return 0

# 单个患者预测函数
def predict_single_patient(input_data, model, scaler, median_imputer, mode_imputer, feature_info):
    try:
        # 转换为DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_info['feature_names'])

        # 数据预处理
        numeric_data = median_imputer.transform(input_df[feature_info['numeric_features']])
        categorical_data = mode_imputer.transform(input_df[feature_info['categorical_features']])

        # 重新组合
        processed_data = np.column_stack([numeric_data, categorical_data])
        processed_df = pd.DataFrame(processed_data, columns=feature_info['feature_names'])

        # 标准化
        scaled_data = scaler.transform(processed_df)

        # 预测
        probability = model.predict_proba(scaled_data)[0][1]
        prediction = model.predict(scaled_data)[0]

        return probability, prediction

    except Exception as e:
        st.error(f"预测过程中出错: {e}")
        return None, None

# 主应用
def main():
    # 标题和介绍
    st.title("👁️ 糖尿病视网膜病变（PDR）风险预测系统")
    st.markdown("---")

    # 加载模型
    with st.spinner("正在加载预测模型..."):
        model, scaler, median_imputer, mode_imputer, feature_info, selected_features = load_model_and_preprocessors()

    if model is None:
        st.error("无法加载模型，请检查模型文件是否存在。")
        return

    # === 核心修改2：侧边栏 - 更新为两级医疗机构层次选择 ===
    st.sidebar.header("🏥 医疗机构层次")
    facility_level = st.sidebar.radio(
        "选择您的医疗机构层次：",
        ["初级（仅基本信息）",  # 对应 basic 组
         "高级（全部指标）"],  # 对应 basic + advanced 组
        index=1  # 默认选中“高级”
    )

    # === 核心修改3：更新层级选择逻辑 ===
    if facility_level == "初级（仅基本信息）":
        selected_groups = ['basic']
    else:  # "高级（全部指标）"
        selected_groups = ['basic', 'advanced']
    # === 核心修改3结束 ===

    # 侧边栏 - 预测模式选择
    st.sidebar.header("🔍 预测模式")
    prediction_mode = st.sidebar.radio(
        "选择预测模式：",
        ["单患者预测", "批量预测"],
        index=0
    )

    # 批量预测时显示文件上传和模板下载
    if prediction_mode == "批量预测":
        st.sidebar.header("📁 批量预测设置")

        # 文件格式选择
        file_format = st.sidebar.radio(
            "选择文件格式：",
            ["Excel (.xlsx)", "CSV (.csv)"],
            index=0
        )

        # 提供模板文件下载
        st.sidebar.markdown("### 📋 数据模板")

        # === 核心修改4：更新批量预测模板生成逻辑 ===
        # 创建模板数据（仅包含当前选中级别的特征）
        template_data = {}
        for group_key in selected_groups:  # 根据用户选择的层级动态生成
            for feature in FEATURE_GROUPS[group_key]['features']:
                # 设置默认值 (此部分逻辑与之前一致，但遍历的组由selected_groups决定)
                if feature == '性别':
                    template_data[feature] = [1]
                elif feature in ['吸烟史', '饮酒史', '高血压病史']:
                    template_data[feature] = [0]
                elif feature == '年龄':
                    template_data[feature] = [50]
                elif feature == '糖尿病病程':
                    template_data[feature] = [5.0]
                elif feature == 'BMI':
                    template_data[feature] = [24.0]
                elif feature == '腰臀比':
                    template_data[feature] = [0.9]
                elif feature == '高血压病程':
                    template_data[feature] = [0.0]
                elif feature in ['收缩压', '舒张压']:
                    template_data[feature] = [120.0, 80.0][['收缩压', '舒张压'].index(feature)]
                elif feature == '血尿素氮':
                    template_data[feature] = [5.0]
                elif feature == '血清肌酐':
                    template_data[feature] = [70.0]
                elif feature == '尿酸':
                    template_data[feature] = [300.0]
                elif feature == '总蛋白':
                    template_data[feature] = [70.0]
                elif feature == '白蛋白':
                    template_data[feature] = [45.0]
                elif feature == '总胆红素':
                    template_data[feature] = [12.0]
                elif feature == '直接胆红素':
                    template_data[feature] = [4.0]
                elif feature == '谷丙转氨酶':
                    template_data[feature] = [25.0]
                elif feature == '谷草转氨酶':
                    template_data[feature] = [26.0]
                elif feature == '空腹血糖':
                    template_data[feature] = [6.5]
                elif feature == '糖化血红蛋白':
                    template_data[feature] = [6.5]
                elif feature == '尿白蛋白排泄率':
                    template_data[feature] = [20.0]
                else:
                    template_data[feature] = [0.0]
        # === 核心修改4结束 ===

        template_df = pd.DataFrame(template_data)

        # 根据选择的文件格式提供不同的模板下载
        if file_format == "Excel (.xlsx)":
            # 生成Excel模板
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='模板')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="pdr预测模板.xlsx">📥 下载Excel模板</a>'
        else:
            # 生成CSV模板
            csv = template_df.to_csv(index=False, encoding='utf-8-sig')
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="pdr预测模板.csv">📥 下载CSV模板</a>'

        st.sidebar.markdown(href, unsafe_allow_html=True)

        st.sidebar.markdown("### 📝 使用说明")
        st.sidebar.info("""
        1. 下载上方模板
        2. 在模板中填写患者数据
        3. 上传填写好的文件
        4. 系统将自动进行批量预测
        5. 下载包含预测结果的文件
        """)

    st.sidebar.header("ℹ️ 关于")
    st.sidebar.info(
        "本系统基于LightGBM机器学习模型，用于预测增殖性糖尿病视网膜病变（PDR）的风险。"
        "请输入患者临床指标以获取风险评估。"
    )

    # === 核心修改5：更新侧边栏显示的模型信息文本 ===
    st.sidebar.header("📊 模型信息")
    st.sidebar.text(f"医疗机构层次：{facility_level}")
    st.sidebar.text(f"预测模式：{prediction_mode}")
    # === 核心修改5结束 ===

    # 根据预测模式显示不同的界面
    if prediction_mode == "单患者预测":
        # 创建两列布局
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("📝 患者信息输入")
            st.markdown(f"**当前层次：{facility_level}**")

            # 创建表单
            with st.form("prediction_form"):
                input_values = {}

                # 根据选择的组显示相应的输入字段
                for group_key in selected_groups:
                    group = FEATURE_GROUPS[group_key]
                    st.subheader(group['name'])

                    # 根据特征数量决定列数
                    features = group['features']
                    if len(features) <= 5:
                        cols = st.columns(len(features))
                        for idx, feature in enumerate(features):
                            with cols[idx]:
                                # 确定特征类型
                                if feature in feature_info.get('categorical_features', []):
                                    feature_type = 'categorical'
                                else:
                                    feature_type = 'numeric'

                                input_values[feature] = create_input_field(feature, feature_type, feature_info)
                    else:
                        # 对于较多特征，使用多行显示
                        num_rows = (len(features) + 2) // 3
                        for row in range(num_rows):
                            row_features = features[row * 3:(row + 1) * 3]
                            if row_features:
                                cols = st.columns(3)
                                for idx, feature in enumerate(row_features):
                                    with cols[idx]:
                                        if feature in feature_info.get('categorical_features', []):
                                            feature_type = 'categorical'
                                        else:
                                            feature_type = 'numeric'

                                        input_values[feature] = create_input_field(feature, feature_type, feature_info)

                # 提交按钮
                submitted = st.form_submit_button("🔍 开始预测", use_container_width=True)

        # 预测结果显示区域
        with col2:
            st.header("📊 预测结果")

            if submitted:
                with st.spinner("正在分析数据..."):
                    # 准备输入数据 - 确保所有特征都有值
                    full_input_data = []
                    for feature in feature_info['feature_names']:
                        if feature in input_values:
                            full_input_data.append(input_values[feature])
                        else:
                            # 对于未输入的字段，使用默认值
                            if feature in feature_info.get('categorical_features', []):
                                full_input_data.append(0)
                            else:
                                full_input_data.append(0.0)

                    # 进行预测
                    probability, prediction = predict_single_patient(
                        full_input_data, model, scaler, median_imputer, mode_imputer, feature_info
                    )

                    if probability is not None:
                        # 显示风险概率
                        st.subheader("风险评估")

                        # 创建仪表盘
                        fig, ax = plt.subplots(figsize=(8, 4))
                        risk_level = "高风险" if prediction == 1 else "低风险"
                        colors = ['#FF4B4B', '#00D4AA']
                        color = colors[1] if prediction == 0 else colors[0]

                        ax.barh([0], [probability * 100], color=color, alpha=0.7)
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('PDR风险概率 (%)')
                        ax.set_yticks([])
                        ax.set_title(f'风险概率: {probability * 100:.2f}%')

                        # 添加风险阈值线
                        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='风险阈值')
                        ax.legend()

                        st.pyplot(fig)

                        # 显示详细结果
                        st.metric(
                            label="预测结果",
                            value=risk_level,
                            delta=f"{probability * 100:.2f}%"
                        )

                        # 显示使用的特征
                        with st.expander("📋 本次预测使用的特征"):
                            used_features = []
                            for group_key in selected_groups:
                                used_features.extend(FEATURE_GROUPS[group_key]['features'])

                            st.write(f"**使用的特征总数: {len(used_features)}**")
                            for i, feature in enumerate(used_features, 1):
                                st.write(f"{i}. {feature}")

                        # 建议信息
                        st.subheader("💡 建议")
                        if prediction == 1:
                            st.error(
                                "⚠️ **高风险预警**:\n\n"
                                "• 建议立即进行详细眼科检查\n"
                                "• 严格控制血糖和血压\n"
                                "• 定期进行眼底检查\n"
                                "• 遵医嘱进行必要的治疗干预"
                            )
                        else:
                            st.success(
                                "✅ **低风险提示**:\n\n"
                                "• 继续保持良好的血糖控制\n"
                                "• 每年进行一次眼科检查\n"
                                "• 保持健康的生活方式\n"
                                "• 如出现视力变化请及时就医"
                            )

                        # 免责声明
                        st.info(
                            "**免责声明**: 本预测结果基于机器学习模型，仅供参考，不能替代专业医学诊断。如有疑问，请咨询医疗专业人员。"
                        )

            else:
                # 默认显示等待信息
                st.info("请在左侧填写患者信息，然后点击'开始预测'按钮")

                # 显示当前层次使用的特征
                with st.expander("📋 当前层次可用的特征"):
                    used_features = []
                    for group_key in selected_groups:
                        group = FEATURE_GROUPS[group_key]
                        st.write(f"**{group['name']}** ({len(group['features'])} 个特征):")
                        for i, feature in enumerate(group['features'], 1):
                            st.write(f"  {i}. {feature}")
                        used_features.extend(group['features'])

                    st.write(f"\n**总特征数: {len(used_features)}**")

    else:  # 批量预测模式
        st.header("📁 批量预测")
        st.markdown(f"**当前医疗级别: {facility_level}**")

        # 文件上传区域
        if file_format == "Excel (.xlsx)":
            uploaded_file = st.file_uploader(
                "上传患者数据Excel文件",
                type=['xlsx', 'xls'],
                help="请确保Excel文件包含必要的特征列。您可以使用左侧的模板文件。"
            )
        else:
            uploaded_file = st.file_uploader(
                "上传患者数据CSV文件",
                type=['csv'],
                help="请确保CSV文件包含必要的特征列。您可以使用左侧的模板文件。"
            )

        if uploaded_file is not None:
            try:
                # 根据文件类型读取数据
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # 读取Excel文件
                    batch_df = pd.read_excel(uploaded_file, engine='openpyxl')
                    st.success(f"✅ 成功读取Excel文件，共 {len(batch_df)} 行数据")
                else:
                    # 读取CSV文件
                    batch_df = pd.read_csv(uploaded_file)
                    st.success(f"✅ 成功读取CSV文件，共 {len(batch_df)} 行数据")

                # 显示数据预览
                st.subheader("📋 数据预览")
                st.write(f"数据形状: {batch_df.shape[0]} 行 × {batch_df.shape[1]} 列")

                # 显示前几行数据
                st.dataframe(batch_df.head(), use_container_width=True)

                # 显示数据统计信息
                with st.expander("📊 数据统计信息"):
                    st.write("**数值型特征统计:**")
                    numeric_cols = batch_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(batch_df[numeric_cols].describe())
                    else:
                        st.write("未找到数值型特征")

                    st.write("**缺失值统计:**")
                    missing_data = batch_df.isnull().sum()
                    missing_df = pd.DataFrame({
                        '特征': missing_data.index,
                        '缺失值数量': missing_data.values,
                        '缺失率 (%)': (missing_data.values / len(batch_df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['缺失值数量'] > 0]
                    if len(missing_df) > 0:
                        st.dataframe(missing_df)
                    else:
                        st.success("✅ 数据完整，无缺失值")

                # 检查必要的特征
                required_features = []
                for group_key in selected_groups:
                    required_features.extend(FEATURE_GROUPS[group_key]['features'])

                missing_features = set(required_features) - set(batch_df.columns)
                if missing_features:
                    st.warning(f"⚠️ 数据中缺少以下必要特征: {list(missing_features)}")
                    st.info("请确保上传的文件包含所有必要的特征列。您可以使用左侧的模板文件。")

                else:
                    # 开始批量预测按钮
                    if st.button("🚀 开始批量预测", type="primary", use_container_width=True):
                        with st.spinner(f"正在对 {len(batch_df)} 名患者进行预测..."):
                            # 执行批量预测
                            results_df = batch_predict(
                                batch_df, model, feature_info, median_imputer, mode_imputer, scaler, selected_features
                            )

                            # 显示预测结果
                            st.subheader("📊 批量预测结果")

                            # 统计信息
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("患者总数", len(results_df))
                            with col2:
                                high_risk_count = (results_df['PDR预测概率'] > 0.7).sum()
                                st.metric("高风险患者", high_risk_count)
                            with col3:
                                medium_risk_count = ((results_df['PDR预测概率'] >= 0.3) & (
                                            results_df['PDR预测概率'] <= 0.7)).sum()
                                st.metric("中风险患者", medium_risk_count)
                            with col4:
                                low_risk_count = (results_df['PDR预测概率'] < 0.3).sum()
                                st.metric("低风险患者", low_risk_count)

                            # 风险分布图
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                            # 风险等级分布
                            risk_counts = results_df['PDR风险等级'].value_counts()
                            colors = ['#00D4AA', '#FFA500', '#FF4B4B']  # 绿、橙、红
                            ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                                    colors=colors[:len(risk_counts)], startangle=90)
                            ax1.set_title('风险等级分布')

                            # 风险概率分布直方图
                            ax2.hist(results_df['PDR预测概率'], bins=20, edgecolor='black', alpha=0.7,
                                     color='steelblue')
                            ax2.axvline(x=0.3, color='orange', linestyle='--', label='低/中风险阈值')
                            ax2.axvline(x=0.7, color='red', linestyle='--', label='中/高风险阈值')
                            ax2.set_xlabel('PDR风险概率')
                            ax2.set_ylabel('患者数量')
                            ax2.set_title('风险概率分布')
                            ax2.legend()
                            ax2.grid(alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig)

                            # 高风险患者详情
                            high_risk_df = results_df[results_df['PDR风险等级'] == '高风险']
                            if not high_risk_df.empty:
                                st.warning(f"⚠️ **发现 {len(high_risk_df)} 名高风险患者**")
                                with st.expander("🔴 高风险患者详情"):
                                    st.dataframe(high_risk_df, use_container_width=True)

                            # 显示结果表格
                            st.subheader("📋 详细预测结果")

                            # 添加筛选功能
                            st.markdown("**筛选预测结果:**")
                            filter_col1, filter_col2 = st.columns(2)
                            with filter_col1:
                                risk_filter = st.selectbox(
                                    "按风险等级筛选:",
                                    ["全部", "高风险", "中风险", "低风险"]
                                )
                            with filter_col2:
                                probability_filter = st.slider(
                                    "按预测概率筛选:",
                                    0.0, 1.0, (0.0, 1.0), 0.01
                                )

                            # 应用筛选
                            filtered_df = results_df.copy()
                            if risk_filter != "全部":
                                filtered_df = filtered_df[filtered_df['PDR风险等级'] == risk_filter]
                            filtered_df = filtered_df[
                                (filtered_df['PDR预测概率'] >= probability_filter[0]) &
                                (filtered_df['PDR预测概率'] <= probability_filter[1])
                                ]

                            st.write(f"筛选结果: {len(filtered_df)} 名患者")
                            st.dataframe(filtered_df, use_container_width=True)

                            # 下载按钮
                            st.markdown("### 💾 下载结果")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            # 提供多种格式下载
                            col1, col2 = st.columns(2)
                            with col1:
                                # Excel格式下载
                                excel_filename = f"pdr预测结果_{timestamp}.xlsx"
                                st.markdown(get_table_download_link(results_df, excel_filename, "excel"),
                                            unsafe_allow_html=True)
                            with col2:
                                # CSV格式下载
                                csv_filename = f"pdr预测结果_{timestamp}.csv"
                                st.markdown(get_table_download_link(results_df, csv_filename, "csv"),
                                            unsafe_allow_html=True)

                            # 单独下载高风险患者
                            if not high_risk_df.empty:
                                st.markdown("#### 🔴 高风险患者单独下载")
                                high_risk_filename = f"pdr高风险患者_{timestamp}.xlsx"
                                st.markdown(get_table_download_link(high_risk_df, high_risk_filename, "excel"),
                                            unsafe_allow_html=True)

            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    st.info("请确保上传的文件是有效的Excel格式，且包含正确的列名。")
                else:
                    st.info("请确保上传的文件是有效的CSV格式，且包含正确的列名。")

        else:
            # 显示批量预测说明
            st.info(f"""
            ## 📝 批量预测使用说明 (使用{file_format})

            1. **下载模板**: 在左侧边栏下载{file_format}模板
            2. **填写数据**: 在模板中填写患者信息（可填写多名患者）
            3. **上传文件**: 使用上方文件上传器上传{file_format}文件
            4. **开始预测**: 点击"开始批量预测"按钮
            5. **查看结果**: 查看统计信息、可视化图表和详细结果
            6. **下载结果**: 预测完成后下载结果文件

            ### 📋 数据要求
            - {file_format}格式文件
            - 包含必要的特征列（根据当前医疗级别）
            - 数值型特征填写数字
            - 分类特征填写0或1（如性别：1=男，0=女）

            ### ⚠️ 注意事项
            - 系统会自动处理缺失值
            - 确保特征单位与模板一致
            - 预测结果仅供参考，不能替代专业诊断
            """)

    # 底部信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "基于LightGBM机器学习模型 | 仅供医疗专业人员参考"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()