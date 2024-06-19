import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

os.environ["TAVILY_API_KEY"] = "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"

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
llm = ChatGroq(
    api_key="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")

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

questions = input("questions:")
agent_executor.invoke({"input": f"{questions} ?用中文回答"})
