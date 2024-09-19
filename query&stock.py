import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
import yfinance as yf

# Load environment variables from a .env file
load_dotenv()

# Ensure the OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The environment variable 'OPENAI_API_KEY' is not set.")

# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'YourCustomUserAgent/1.0'

# Load documents from the web
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Create FAISS vector store from documents
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
print(retriever)

# Initialize Wikipedia API Wrapper and Query Run
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
print(wiki.name)

# Create a retriever tool for LangSmith search
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")
print(retriever_tool.name)

# Initialize Arxiv API Wrapper and Query Run
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)

# Define a custom function for stock price
def get_stock_price(ticker):
    """
    Fetches and returns the current stock price of the given ticker.
    
    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
    :return: Current stock price or error message if ticker is invalid.
    """
    try:
        # Download the stock data for the given ticker
        stock = yf.Ticker(ticker)
        
        # Get the live stock price
        stock_info = stock.history(period='1d')
        current_price = stock_info['Close'][0]  # Get the most recent closing price

        return f"Current stock price of {ticker}: ${current_price}"
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"

# Create a custom tool for stock price
stock_tool = Tool(
    name="stock_tool",
    func=lambda query: get_stock_price(query),
    description="Fetches the current stock price of a given ticker symbol."
)

# Create a list of tools
tools = [wiki, arxiv, retriever_tool, stock_tool]
print(tools)

# Initialize ChatOpenAI with specific model and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=openai_api_key)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent", api_key=openai_api_key)
print(prompt.messages)

# Create an OpenAI tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an AgentExecutor with the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor)

# Debugging: Print the list of tools and their descriptions
for tool in tools:
    print(f"Tool name: {tool.name}, Description: {tool.description}")

# Continuous loop to prompt user for queries
while True:
    query1 = input("Enter the query1 to search (use Wikipedia, Arxiv tool) or type 'exit' to quit: ")
    if query1.lower() == 'exit':
        break
    response_wiki_arxiv = agent_executor.invoke({"input": query1})
    print("Response from Wikipedia/Arxiv tool:", response_wiki_arxiv)

    query2 = input("Enter the stock ticker symbol (use stock tool) or type 'exit' to quit: ")
    if query2.lower() == 'exit':
        break
    response_stock = stock_tool.func(query2)  # Directly call the stock tool function
    print("Response from stock tool:", response_stock)