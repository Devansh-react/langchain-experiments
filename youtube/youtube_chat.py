from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
#creates a transcript of the given youtube URL  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
# to manuplate the no of token limit  we use text splitter , basically to use OPEN AI api 
    db = FAISS.from_documents(docs, embeddings)
    # embeddings convert all to those strings or text to number then FAISS is the library that peforms silmarity search for that doc we have just created
    return db
# then return the desired database 

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
#calculates the similarity between the query vector and the document vectors stored in the database. It returns the k most similar documents. k defaults to 4, meaning the top 4 most similar documents are retrieved.
    docs_page_content = " ".join([d.page_content for d in docs])
#a single string containing the combined text content of those documents. This string is then typically used to provide context to a large language model (LLM) for answering the user's query.
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)
#ai model initalised using chat based interface 




    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
#takes a list of message prompt templates (in this case, the system and human messages) and assembles them into a structured chat prompt understood by the LLMChain.

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")       
    return response, docs


# Example usage:
video_url = "https://www.youtube.com/watch?v=th4j9JxWGko"
db = create_db_from_youtube_video_url(video_url)

query = "what is this video about?"
response, docs = get_response_from_query(db, query)
print(textwrap.fill(response, width=50))
