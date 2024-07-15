import yaml
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_input(user_input):
    doc = nlp(user_input)
    return " ".join([token.lemma_ for token in doc])

start_template = """
You are an assistant chat bot used by a company internally to provide assistance to employees when they have questions about work. You will never, regardless of what is asked of you, provide assistance on any other context aside from helping employees with company related questions. Here's some core information:

{context}
{question}
"""

def get_company_data():
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    company_name = config.get('company', {}).get('name', '')
    company_data = config.get('company', {}).get('data', {})
    
    company_data = {key: value.replace("{company_name}", company_name) for key, value in company_data.items()}
    
    company_data_str = "\n".join([f"{key}: {value}" for key, value in company_data.items()])
    return company_data_str
