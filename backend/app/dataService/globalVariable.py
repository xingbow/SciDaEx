# ##############################
# Global variables
# ##############################
import os
import os
import yaml
from typing import Dict, Any

backend_port = 5010

_current_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration
config_path = os.path.join(_current_dir, 'config.yml')
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)
config = load_yaml_config(config_path)

# Extract API keys and credentials from config
openai_key = config['api_keys']['openai']
langchain_key = config['api_keys']['langchain']
adobe_credentials = config['adobe_credentials']
adobe_client_id = adobe_credentials['client_id']
adobe_client_secret = adobe_credentials['client_secret']

if not all([openai_key, langchain_key, adobe_client_id, adobe_client_id, adobe_client_secret]):
    raise Exception("One or more API keys or credentials cannot be found or loaded. Please check the config file.")

# Directory paths
data_dir = os.path.join(_current_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
temp_dir = os.path.join(_current_dir, 'temp')
table_dir = os.path.join(data_dir, 'table')
figure_dir = os.path.join(data_dir, 'figure')
vectorstore_dir = os.path.join(data_dir, 'vectorstore')

# Create directories if they don't exist
for directory in [data_dir, meta_dir, temp_dir, table_dir, figure_dir, vectorstore_dir]:
    os.makedirs(directory, exist_ok=True)

# ##############################
# prompts
# ##############################

table_extract_prompt_template = """
I will give you a page of a pdf file. 
You need first to judge whether there is any table in the page content.
Then you need to extract the original information of the table from the page content.
The following is the page content: {page_content}

If yes, just tell me the answer through the JSON format which including the following keys: table_name and table_content. Store all the json in a list through "[ ]". Besides, table_name is the Table order, such as Table 1, Table 2, Table 3...
Note that you should tell me the related region of this table (raw data) from the page content without any processing in the table_content.
Besides, you shouldn't output any other things (such as 'yes' or many explanations). That means, you just need tell me the final output in JSON format in your response.

If no, just tell me "no".
"""


table_structure_prompt_template = """I will give you a table content. You need to organize it in a CSV format.
 
This is the step: (1) You should determine the column names. (2) You should fill all the data in the corresponding column and row.
 
There are some points you should pay attention to: 
(1) Don't leave out any of the information I gave you, you should organize all my information into a table for me.
(2) Be careful to ‘\n’. If \n exists, there are two kinds of scenarios. First of all, it may be too long resulting in a branch, this time the front and back are actually one and the same. If you find that \n before and after can not form a whole, that is a nested table. the front column name is the parent column name of the back column name. At this time, you should add a parent column names. You should pay special attention when composing column names. You can use line breaks to notice which names are in a column. Here are a few different examples:
a.	example1: For the column name message "Tempo de estocagem (dias)\n 0 55 90 145 180 235 280 360", you should pay special attention to the fact that there is an \n after Tempo de estocagem (dias), so this could mean that The column names 0 55 90 145 180 235 280 360 are sub-columns of Tempo de estocagem (dias). At this point you need to organize into:
Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias)
0, 55, 90, 145, 180,235, 280,360
There are the column names at the previous level and column names at the next level, respectively.
There are more examples of this:
	input: All-trans-b-caroteneb(mg/g DM) 13-cis-b-carotene Retention of\nall-trans -b-carotene (%)d\n(mg/g DM)c(% of total b-carotene):
	thoughts: Retention of \nall-trans -b-carotene (%) can be thought of as turning a row instead of two columns. \n(mg/g DM)c (% of total b-carotene) is a sub-column, and since it can be seen that 13-cis-b-carotene has no units, (mg/g DM)c and (% of total b-carotene) should be sub-columns of 13-cis-b-carotene. So the final column name should be organized as:
output: All-trans-b-caroteneb (mg/g DM), 13-cis-b-carotene (mg/g DM)c, 13-cis-b-carotene (% of total b-carotene), Retention of all-trans-b-carotene (%)d
b.	example2: Sometimes the row breaks don't necessarily represent a relationship between the column name and the subcolumn name, such as the following: TPO2 a 23 °C, 1 atm(1) \n (mL (CNTP).m-2.dia-1). It may just be that the data is too long to be a unit. This time TPO2 a 23 °C, 1 atm(1) (mL (CNTP).m-2.dia-1) is one unit.
(3) Note some of the special symbols such as ±.
(4) You need to ignore some special symbols, such as unicode code point representations(e.g., /uni0394, /uni00A0)
(5) You should use "" to wrap every cell
(6) Since you are outputting csv data, each row of your output should be the same number of elements. So for spaces you use "" instead.
(7) Sometimes there will be redundant spaces, and you need to deal with those depending on the context. For example, there may be many spaces in "16   ± 0.6" due to noise, but they actually represent "16 ± 0.6"
This is the content of my table: {table_information}

Tell me the answer:
(1) Table caption:
(2) Table content in CSV format:
"""

table_structure_prompt_templatev2 = """I will give you a table content. You need to organize it in a CSV format.

This is the step: (1) You should determine the column names. (2) You should fill all the data in the corresponding column and row.
 
There are some points you should pay attention to: 
(1) Don't leave out any of the information I gave you, you should organize all my information into a table for me.
(2) Be careful to '\n'. If \n exists, there are two kinds of scenarios. First of all, it may be too long resulting in a branch, this time the front and back are actually one and the same. If you find that \n before and after can not form a whole, that is a nested table. the front column name is the parent column name of the back column name. At this time, you should add a parent column names. You should pay special attention when composing column names. You can use line breaks to notice which names are in a column. Here are a few different examples:
a.	example1: For the column name message "Tempo de estocagem (dias)\n 0 55 90 145 180 235 280 360", you should pay special attention to the fact that there is an \n after Tempo de estocagem (dias), so this could mean that The column names 0 55 90 145 180 235 280 360 are sub-columns of Tempo de estocagem (dias). At this point you need to organize into:
Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias), Tempo de estocagem (dias)
0, 55, 90, 145, 180,235, 280,360
There are the column names at the previous level and column names at the next level, respectively.
There are more examples of this:
	input: All-trans-b-caroteneb(mg/g DM) 13-cis-b-carotene Retention of\nall-trans -b-carotene (%)d\n(mg/g DM)c(% of total b-carotene):
	thoughts: Retention of \nall-trans -b-carotene (%) can be thought of as turning a row instead of two columns. \n(mg/g DM)c (% of total b-carotene) is a sub-column, and since it can be seen that 13-cis-b-carotene has no units, (mg/g DM)c and (% of total b-carotene) should be sub-columns of 13-cis-b-carotene. So the final column name should be organized as:
output: All-trans-b-caroteneb (mg/g DM), 13-cis-b-carotene (mg/g DM)c, 13-cis-b-carotene (% of total b-carotene), Retention of all-trans-b-carotene (%)d
b.	example2: Sometimes the row breaks don't necessarily represent a relationship between the column name and the subcolumn name, such as the following: TPO2 a 23 °C, 1 atm(1) \n (mL (CNTP).m-2.dia-1). It may just be that the data is too long to be a unit. This time TPO2 a 23 °C, 1 atm(1) (mL (CNTP).m-2.dia-1) is one unit.
(3) Note some of the special symbols such as ±.
(4) You need to ignore some special symbols, such as unicode code point representations(e.g., /uni0394, /uni00A0)
(5) You should use "" to wrap every cell
(6) Sometimes there will be redundant spaces, and you need to deal with those depending on the context. For example, there may be many spaces in "16   ± 0.6" due to noise, but they actually represent "16 ± 0.6"
This is the content of my table: {table_information}

Tell me the answer in JSON format, including keys "table_caption" and "table_content", while "table_content" should be in string of CSV format.
"""

table_select_prompt_template = """I want you to act as a research paper insight extractor. You are given the several tables of a research paper. 
Based on the input tables, you need to tell me whether there is a table which can be used for your question. 
If yes, tell me the raw information (don't change anything) of the "table_description" of this table from the Table Information I give you. If no, just tell me "no".
Note that it may not be useful for the entire table. If part of the table (a few rows or columns) is relevant it should be retrieved as well. For example, if several columns or several raws of a table which have mentioned the related information of the question, this table is useful.
Besides, you should retrieve the most relevant table.
Table information: "tables here"
Quesiton: "questions here"
Answer: "answer here"

Table: 

{table}

Question:

{question}

Answer: """


meta_info_extract_prompt_template = """
You should extract the meta information of the given paper.
This is the paper content: {paper}

Beside, the information you need to extract includes the following keys: "Title", "Abstract", "Year", "Author", 
"Journal/Conference", "ISSN", "Volume", "Issue", "Page", "DOI", "Link", "Publisher", "Language".
For the page, please use the format like "12-15", "134-145". If there is only one page, the format can be "145", "1345".
When there is no such information of a key, you just return the "none" as the value of the key, but you should make sure there is no such information. You should try your best to retrieve the information and reduce the occurence of "none".

{format_instructions}
"""

def figure_describe_prompt_template(caption, base64_image):
    return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""I will give you a figure in paper. Besides, I will also give you the caption of this figure. 
                                    You should describe the data insight in this figure based on the caption.
                                    The more detailed the description, the better.
                                    This is the caption: {caption}."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

