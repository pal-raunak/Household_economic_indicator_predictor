
import os
import yaml
import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import tiktoken


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(funcName)s:%(lineno)d: %(message)s",
    handlers=[
        logging.FileHandler("./profile.log"),
        logging.StreamHandler()
    ]
)

# AzureOpenAI LLM class
class UtilsAzureOpenAI:
    def __init__(self, 
                 config_file="app/config/server.yml",
                 temperature=0):
        
        azure_end_point    = "",
        azure_openai_key   = "",
        azure_deployment   = ""
        openai_api_version = ""
        self.con_OK        = True

        try:
            with open(config_file) as stream:
                _yaml   = yaml.safe_load(stream)

                current_model      = _yaml['profiling']['llm']['model_name']
                embedding_model    = _yaml['profiling']['llm']['embedding_model']
                

                azure_end_point    = _yaml['profiling']['llm'][current_model]['azure_end_point']
                azure_openai_key   = _yaml['profiling']['llm'][current_model]['azure_openai_key']
                azure_deployment   = _yaml['profiling']['llm'][current_model]['azure_deployment']
                openai_api_version = _yaml['profiling']['llm'][current_model]['openai_api_version']

                self.cost_per_1000_input_tokens = _yaml['profiling']['llm'][current_model]['cost_per_1000_input_tokens']
                self.cost_per_1000_output_tokens = _yaml['profiling']['llm'][current_model]['cost_per_1000_output_tokens']

                con_url = _yaml['profiling']['neo4j']['con_url']
                auth    = (_yaml['profiling']['neo4j']['user'], _yaml['profiling']['neo4j']['passwd'])

                embed_openai_key        = _yaml['profiling']['llm'][embedding_model]['azure_openai_key']
                embed_openai_endpoint   = _yaml['profiling']['llm'][embedding_model]['azure_end_point']

                logger.debug(con_url)
                logger.debug(auth)
        except Exception as e:
                logger.error(e)
                print(e)
                self.con_OK = False
        
        try:
            self.llm = AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_end_point,
                api_key=azure_openai_key,
                temperature=temperature,
                )
            
            self.llm_gpt4 = AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment='gpt-4o',
                azure_endpoint='https://openai4352456436436.openai.azure.com/',
                api_key='ae7414894ce54e9cb6f8c21472f8fbba',
                temperature=temperature,
                )
            
            self.embd = AzureOpenAIEmbeddings(model=embedding_model,
                                              openai_api_version=openai_api_version,
                                              api_key=embed_openai_key,
                                              azure_endpoint=embed_openai_endpoint)
            
            self.tiktoken = tiktoken.encoding_for_model(current_model)
        except Exception as e:
             logger.error(e)
             self.con_OK = False

        if not self.test():
            self.con_OK = False
    
    def get_llm(self):
       return self.llm
    
    def get_embd(self):
       return self.embd
    
    def resource_calculation(self, prompt, response):
        input_token_count  = len(self.tiktoken.encode(prompt))
        output_token_count = len(self.tiktoken.encode(response))

        total_tokens = input_token_count + output_token_count
        logger.debug(f"Total token count: {total_tokens}")

        # Calculate costs
        input_cost  = (input_token_count / 1000)  * self.cost_per_1000_input_tokens
        output_cost = (output_token_count / 1000) * self.cost_per_1000_output_tokens
        total_cost  = input_cost + output_cost
        logger.info(f"Total cost: {input_token_count}, {output_token_count}")
        logger.debug(f"Total cost: {total_cost}")

        return (total_tokens, round(total_cost, 5))
    
    def resource_calculation_gpt4(self, prompt, response):
        input_token_count  = len(self.tiktoken.encode(prompt))
        output_token_count = len(self.tiktoken.encode(response))

        total_tokens = input_token_count + output_token_count
        logger.debug(f"Total token count: {total_tokens}")

        # Calculate costs
        input_cost  = (input_token_count / 1000)  * 0.0025
        output_cost = (output_token_count / 1000) * 0.0100
        total_cost  = input_cost + output_cost
        logger.debug(f"Total cost: {total_cost}")

        return (total_tokens, total_cost)
    
    def invoke(self, prompt, resource_needed=False):
        resource = (0.0, 0.0)
        response = self.llm.invoke(prompt).content

        if resource_needed:
            resource = self.resource_calculation(prompt, response)
            logger.warning("Resource Utilized : {}".format(resource))

        return response, resource
    
    def invoke_gpt4(self, prompt, resource_needed=False):
        resource = (0.0, 0.0)
        response = self.llm_gpt4.invoke(prompt).content

        if resource_needed:
            resource = self.resource_calculation_gpt4(prompt, response)
            logger.info("$$$$ Resource: {}".format(resource))

        return response, resource
    
    def test(self, text="Hi"):
        try:
            res = self.llm.invoke(text)
            return True
        except Exception as e:
            logger.error(e)
            return False
         
    