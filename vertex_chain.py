# Dependencies for usage example.
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from wrapper.vertex_wrapper import AllChainDetails
import vertexai

import os

project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
model_name = os.getenv("MODEL_NAME")

def call_llm()-> None: 
    vertexai.init(project=project_id, location=location)
    llm = VertexAI(model_name=model_name, temperature=0)

    # Callback handler specified at execution time, more information given.
    prompt_template = "What food pairs well with {food}?"
    handler = AllChainDetails()
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    llm_chain("chocolate", callbacks=[handler])

if __name__ == '__main__':
    call_llm()


