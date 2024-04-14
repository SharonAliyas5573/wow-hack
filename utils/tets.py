# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("utils/sodapdf-converted-1.pdf")
# documents = loader.load_and_split()

# from ai_utils import get_summarization_prompt, get_response

# prompt = get_summarization_prompt(history=documents[0].page_content)

# print(prompt)

# res = get_response(prompt)

# print(res.content[0].text)
# print("\n\n------------\n\n")
# print(res)


from db_helpers import create_patient_record, query_patient_record
from ai_utils import *


res = query_patient_record("test")
res = " ".join(str(res) for res in res)
# print(res)

# ades = get_ade_response(res)
# ades = " ".join(str(ades.page_content) for ades in ades)
# print(ades)

ade_prompt = get_recommendation_prompt(
    context=res,
    article="test",
    symptoms="The patient has trouble sleeping and has a weird rash on their arm.",
)
# print(ade_prompt)

res = get_response(ade_prompt)
print("\n\n")
print(res.content[0].text)

# create_patient_record("test", "utils/sodapdf-converted-1.pdf")
