import os

from anthropic import Anthropic
from dotenv import load_dotenv
from langchain.vectorstores import Chroma


from .docs_helpers import embeddings

load_dotenv()

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

RECOMMENDATION = """
Given below is the history of a patient, i need you to keep this in mind while answering the questions.
context: {}
These are some articles that you can refer to:
article: {}
                  
Now iam going to give you the current symptoms of the patient.
symptoms: {}
Now with the above context and symptoms, can you please provide some helpful diffrential diagnosis and any leads for further investigation, also provide what information lead you to this conclusion. and also provide the treatment recommendation. and some additional information that you think is important with some references."""

SUMMARIZATION = """
Given below is the history of a patient, write a comprehensive summary of the patients history. Also be sure to show stats like blood pressure etc as a colon sperated list.
history: {}"""

ADE = """
I provide some real world events that are experienced by patients due to the drugs they are taking.
ade: {}
Now iam going to give you the patient's history, can you please provide the adverse drug event that the patient could experiencing.
history: {}"""


def get_ade_prompt(ade, history):
    return ADE.format(ade, history)


def get_summarization_prompt(history):
    return SUMMARIZATION.format(history)


def get_recommendation_prompt(context, article, symptoms):
    return RECOMMENDATION.format(context, article, symptoms)


def get_ade_response(query):
    vectordb = Chroma(persist_directory=f"./ade", embedding_function=embeddings)

    matching_docs = vectordb.similarity_search(query)

    return matching_docs


def get_article_response(query):
    vectordb = Chroma(persist_directory=f"knowledge/", embedding_function=embeddings)

    matching_docs = vectordb.similarity_search(query)

    return matching_docs


def get_response(prompt):
    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-3-haiku-20240307",
    )
    return message
