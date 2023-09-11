import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import altair as alt
import base64
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import openai
import os
from langchain.llms import AzureOpenAI  # 使用了azure
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler


# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["OPENAI_API_BASE"] = "https://oh-ai-openai-scu.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "c33ce426568e41448a5f942ec58a4bda"
# start_phrase = 'Tell me a joke.'

# response = openai.Completion.create(engine=deployment_name, model="gpt-35-turbo", prompt=start_phrase, max_tokens=10)
# text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
# print(start_phrase+text)



openai_api_key  ="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"

# os.environ["OPENAI_API_KEY"] ="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"

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

def multiplier(a, b):
    return a * b
 
def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))
 


def upload_excel(string):
        if '分析Excel' in string  :
                uploaded_file = st.file_uploader("请选择文件(可多个)：", \
                        accept_multiple_files = False, type=["xlsx","xls"])
                if uploaded_file is not None:
                        df = pd.read_excel(uploaded_file.read())
                        st.dataframe(df)
        # elif '数据清洗' in string  :
        

tools = [
    Tool(
        name="Multiplier",
        func=parsing_multiplier,
        description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2.",
    ),
    
    Tool(
        name="upload_excel",
        func= upload_excel,
        description="When you want to analysis a file of excel, please upload the file.",
    )
    
    
]




with st.form('my_form'):
    text = st.text_area('What can I help you?:', '我想分析Excel')
    submitted = st.form_submit_button('Submit')
    if submitted :
        # generate_response(text)
        st.info( '好的，正在查阅功能' )
        st.info( generate_response(text) )
        # llm =  openai.ChatCompletion.create(model="gpt-3.5-turbo", 
        #                             openai_api_key=  openai_api_key ,#openai.api_key #'openai_api_key,
        #                             openai_api_base = "https://api.xiaojuan.tech/v1",
        #                             messages=[
        #                                 {"role": "system", "content": "您是一个帮助翻译英文到法文的助手。"},
        #                                 {"role": "user", "content": "将以下英文文本翻译成法语：'{}'"} ],     
        #                                     temperature=0)
     
        # agent  = initialize_agent(
        #         tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        # agent.run(text)



        if '分析Excel' in   text :
            #  可添加accept_multiple_files = False类似设置运行上传多Excel文件
            uploaded_file = st.file_uploader("Choose a Excel file",  type=["xlsx","xls"] )   


    
uploaded_file = st.file_uploader("Choose a Excel file",  type=["xlsx","xls"] )

if uploaded_file is not None:    
    df = pd.read_excel(uploaded_file.read())
    st.dataframe(df)                
                
# dataFrame =df
selector = st.multiselect("你想实现哪些功能", ['清洗数据', '特征工程', ' 因子分析', ' 画图'])
st.write(selector)
print('what is your nam')
select_reslt = []


# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# dfcancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# st.dataframe(dfcancer) 
    


for item in selector:
    print(item)
    # if "清洗数据" in item.key():
    if "清洗数据" in item: # 
        dfclear = df.isnull().sum().sort_values(ascending=False) 
        st.write( df.isnull().sum().sort_values(ascending=False)   )  # 统计带有空值的属性信息
        
        dfclear  =  df.dropna(axis=0, how='any', subset=None, inplace=False) #删除带有空值的整行， axis=1,改为按列删除
        # data.fillna(value=0)    #用0填充空值
        st.info("清洗后数据")
        st.dataframe(dfclear) 
        
    if "特征工程" in item:  # 仅进行了离散化
        bins = [-2, 0,  0.5,1.0, 1.5, 3.0]
        df_r = pd.cut( dfclear["attribute 6"],  bins, labels=[1,2,3,4,5] )  # 对该列进行离散化
        st.write( df_r)
        
        fig = plt.figure()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.hist(dfclear["attribute 6"], bins=6)
        st.pyplot()
        
    
    
    if "画图" in item:
                
        n_bin     = dfclear['class'].value_counts()
        print(n_bin)
        class_id  = list(set(dfclear['class']))
        
        labels  = [1,2,3,4,5]
        

        # 可调其它图 ，或用户语言提示生成选项，选择进行画图
        fig = plt.figure()
        plt.pie(n_bin,  labels= class_id, autopct='%1.1f%%',
        shadow=True, startangle=90)#
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    if "因子分析" in item:  # 仅充分性检验 ：Bartlett's Tsest-可添加主成分分析等
        
        df_sd = StandardScaler().fit_transform(dfclear) 
        
        chi_square_value, p_value = calculate_bartlett_sphericity( df_sd )
        print(chi_square_value, 'chi_square_value')
        print(p_value,'p_value')
        st.info('Bartlett Test:chi_square_value, p_value')
        st.info( [chi_square_value, p_value] )
        
        
        kmo_all,kmo_model=calculate_kmo(df_sd)
        st.info('KMO Test: kmo_model' )
        st.info(kmo_all)
        st.info( kmo_model )
        N = len( kmo_all )
        col = []
        for i in range(N):
            col.append("factor" + str(i + 1))
        dfclear.columns = col
        # df_sd.index = f_corr.columns
        
        faa = FactorAnalyzer(5,rotation=None)  #不旋转，
        faa.fit(df_sd)

        ev,v = faa.get_eigenvalues()


        # 可视化
        # plot横轴是指标个数，纵轴是ev值
        # scatter横轴是指标个数，纵轴是ev值
        fig = plt.figure()
        plt.scatter(range(1,df_sd.shape[1]+1),ev)
        plt.plot(range(1,df.shape[1]+1),ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()
        st.pyplot()
        
        faa.loadings_
        df_cm = pd.DataFrame(np.abs(faa.loadings_), index=  dfclear.columns)
        fig = plt.figure()
        # fig,ax = plt.subplots(figsize=(12,10))
        # sns.heatmap(df_cm,annot=True,cmap='BuPu',ax=ax)
        # # 设置y轴字体的大小
        # ax.tick_params(axis='x',labelsize=15)
        # ax.set_title("Factor Analysis",fontsize=12)
        # ax.set_ylabel("Attribute ")


        # plt.figure(figsize=(14, 14))
        # ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
        # ax.yaxis.set_tick_params(labelsize=15) # 设置y轴字体大小
        # plt.title("Factor Analysis", fontsize="xx-large")
        # plt.ylabel("variable", fontsize="xx-large")# 设置y轴标签
        # plt.show()# 显示图片
        # st.pyplot()
        
        plt.figure(figsize = (14,14))
        ax = sns.heatmap( df_cm, annot=True, cmap="BuPu")

        # 设置y轴字体大小
        ax.yaxis.set_tick_params(labelsize=15)
        plt.title("Factor Analysis", fontsize="xx-large")

        # 设置y轴标签
        plt.ylabel("Sepal Width", fontsize="xx-large")
        # 显示图片
        plt.show()
        st.pyplot()

