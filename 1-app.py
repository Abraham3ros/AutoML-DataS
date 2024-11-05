## Auto ML 
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd 
import re
from langchain_openai import ChatOpenAI
import subprocess
import ast

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

## LangSmith Tracking 
os.environ['LANGCHAIN_API_KEY']= "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"
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="AutoML-DataScientist"


## set up Streamlit 
st.title("Explain your problem and give me the data, I'll work my magic on it 😉")

#st.write("Upload your data here")

## Input the API Key
api_key=st.text_input("Enter your API key:",type="password")

## Check if the Api key is provided
if api_key:

    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-70b-versatile")

    error_llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.2-90b-vision-preview")
    #error_llm=ChatOpenAI(model="gpt-4o")

    ## Upload the CSV
    st.write('Make sure the dependent variable or the prediction variable is the last column in the dataset')
    
    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_file=st.file_uploader("Upload your data here:",type="csv",accept_multiple_files=False)
    ##Prompt for the LLM

    prompt= ChatPromptTemplate.from_messages(
        [ ("system", """
            You are an expert Data Scientist capable of writing perfect code and providing clear, understandable explanations. 
            You will be given the header row and the first 5 elements of each column of a dataset. 
            The header row contains all the features, with the last one being the dependent variable (target variable). 
            Your tasks are as follows: - Perform feature engineering: analyze the features to decide which columns are important and which are not. - Decide on appropriate preprocessing steps for the dataset, including handling missing values, encoding categorical variables (e.g., label encoding, one-hot encoding), scaling numerical features, etc. - Handle missing values appropriately. - After preprocessing, split the data into X_train, X_test, y_train, y_test. - Scale X_train and X_test using an appropriate scaler (e.g., StandardScaler), and name the scaled datasets as X_train_scaled and X_test_scaled.
            Your response should be structured as follows: 1. Feature Selection and Model Recommendations - Provide a brief explanation of the features you selected and why. - Explain how the data is processed, including handling of missing values and scaling. 2. Data Preprocessing Code - Provide the code to load and preprocess the data, up to and including the creation of X_train_scaled and X_test_scaled. - Do not include any comments in the code or give an explanation. - Ensure the filename for read_csv() is 'Data.csv'. 3. Suited Models: - List the models that are best suited for this kind of dataset and explain why. 4. State the problem (classification or regression) in exactly the following manner: Problem_Model: Regression or Classification"""), 
            ("user", "{input}") ]
            #("user", "Here is the header row containing the features: {input}") ]
    )

    error_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are an expert Python developer and code error resolver specializing in machine learning implementations and data processing. You will be provided with Python code and the corresponding error message that occurs when running the code. Your task is to:

            - **Identify** the root cause of the error.
            - **Provide the corrected code** that resolves the error.
            - **Ensure** that the sole output of the code is the array of accuracies with the model names, printed using `print(accuracy)`.
            - **Maintain** the overall structure of the original code, especially the code for the models, unless changes are necessary to fix the error.
            - **Include** all necessary imports and handle any missing dependencies.
            - **Avoid** adding explanations or comments; only provide the corrected, runnable code.

            **Important Notes:**

            - The code will always involve ML model implementation and its data processing.
            - The data processing code may change, but the code for the models will remain the same.
            - The corrected code should be error-free and ready to run in a local environment.
            """),
            ("user", "{input}")
        ]
    )

    # Code extractor
    def extract_code(text):
            code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
            return '\n'.join(code_block.strip() for code_block in code_blocks)
    
    def extract_model_type(text):
        match = re.search(r'Problem_Model:\s*(\w+)', text)
        if match:
            return match.group(1)  # Returns the captured group the word following "Problem_Model:"
        return None 
    

    # Get the code for the Classification Models
    def classification_models(filename="4-ClassificationModels.py"):
        # Read the content of the file
        try:
            with open(filename, "r") as file:
                existing_content = file.read()
        except FileNotFoundError:
            existing_content = '' 
        return existing_content
    
    # Get the code for the Regression Models
    def regression_models(filename="3-RegressionModels.py"):
        # Read the content of the file
        try:
            with open(filename, "r") as file:
                existing_content = file.read()
        except FileNotFoundError:
            existing_content = '' 
        return existing_content
    
    # Create a file that contains the Data Preprocessing code given by LLM + The models to Apply based on the problem        
    def create_final_python_file(llm_gen_code,problem_model, filename="final_code.py"):

        #Data Preprocessing Code:
        data_preprocessing_code = llm_gen_code 
        
        models = None

        if problem_model=="Regression":
            models = regression_models()
        elif problem_model=="Classification":
            models = classification_models()

        # Combine ( LLM Code + Models)
        if models:
            with open(filename, "w") as file:
                file.write(data_preprocessing_code + models)
        else:
            st.write("Problem Determining the Model")

    def renew_final_file_after_error(code,filename="final_code.py"):
        with open(filename,'w') as file:
            file.write(code)     

    def get_best_accuracy_model(model_accuracy_string):

        # Convert into a list
        output_list = ast.literal_eval(model_accuracy_string)

        model_accuracies = []

        # Process each item in the list
        for item in output_list:

            model_name, accuracy_str = item.split(':(')
            # Closing parenthesis and convert accuracy to float
            accuracy = float(accuracy_str.rstrip(')'))
            model_accuracies.append((model_name, accuracy))

        best_model = max(model_accuracies, key=lambda x: x[1])

        #print("Model with the highest accuracy:", best_model[0])
        #print("Accuracy:", best_model[1])
        return (best_model[0])



    
    if uploaded_file:
        ds=pd.read_csv(uploaded_file)
        header_row_features=ds.head()
        chain=prompt|llm
        response=chain.invoke({'input':header_row_features})
        st.write(response.content)
        problem_model=extract_model_type(response.content)
        llm_gen_code = extract_code(response.content) 
        create_final_python_file(llm_gen_code,problem_model)
        #Save the DataSet File:
        save_path = "./Data.csv"    
        ds.to_csv(save_path)

        #Executing Final Code and getting the result:
        code_result=subprocess.run(['python','final_code.py'],capture_output=True,text=True)
        output=code_result.stdout
        error=code_result.stderr
        # if Final Code contains error then rerun it through the llm model with error prompt chain, rerun the code and get the output
        if error:
            #st.write(error) Debugging Shows the error
            with open("final_code.py", "r") as file:
                error_code = file.read()

            error_chain=error_prompt|error_llm
            response_after_error=error_chain.invoke({'input':"I am getting this error when trying to run the following code "+error_code+" This is the error "+error})
            st.write(response_after_error) #if error then show the response given by the error_llm

            error_llm_gen_code = extract_code(response_after_error.content) 
            renew_final_file_after_error(error_llm_gen_code)

            error_code_result=subprocess.run(['python','final_code.py'],capture_output=True,text=True)
            new_output=error_code_result.stdout
            #new_error=error_code_result.stderr
            #st.write(new_error) If you want to show the process of debugging the code failed 
            st.write(new_output)
            best_model=get_best_accuracy_model(new_output)
            st.write("The best model with the highest accuracy is "+best_model)
        else:
            st.write(output)
            best_model=get_best_accuracy_model(output)
            st.write("The best model with the highest accuracy is "+best_model)
        
else:
    st.warning("Please enter your API Key")




from langchain.text_splitter import RecursiveCharacterTextSplitter








