
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableConfig
)
from chatapp.llm_tools import  tools

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant specializing in data analysis with a focus on electricity prices and weather data. "
            "Follow these steps to fulfill user requests:\n"
            "1. **Determine Data Requirements**: Understand the user's query and identify the type of data needed (e.g., electricity prices, weather data).\n"
            "2. **Check Data Availability**: Use the SQL tools to check if the required data is already available in the database.\n"
            "   - If data is available, fetch it using the SQL tools.\n"
            "   - If data is not available, use the appropriate fetching tools to obtain the data (e.g., electricity_price_tool, weather_data_tool).\n"
            "3. **Data Analysis**: Once data is obtained, determine the type of statistical analysis required by the user (e.g., descriptive statistics, correlation, regression).\n"
            "   - Use the descriptive_stats_tool for basic statistical summaries.\n"
            "   - Use the correlation_tool to analyze relationships between variables.\n"
            "   - Use the regression_tool for predictive modeling and to understand relationships between variables.\n"
            "4. **Provide Insightful Responses**: Offer concise, data-driven insights and visualizations as needed based on the analysis results. Always ensure your responses are clear, precise, and aligned with the user's query.",
        ),
        ("placeholder", "{messages}"),  
    ]
)


assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)


def create_tool_node_with_fallback(tools):
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state):
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            AIMessage(
                content=f"Error: {repr(error)}. Please fix your mistakes.",
                tool_call_id=tc["id"]
            ) for tc in tool_calls
        ]
    }

tool_node_with_fallback = create_tool_node_with_fallback(tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        """
        Initialize the Assistant with a runnable object.

        Args:
            runnable (Runnable): The runnable instance to invoke.
        """
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        """
        Call method to invoke the LLM and handle its responses.
        Re-prompt the assistant if the response is not a tool call or meaningful text.

        Args:
            state (State): The current state containing messages.
            config (RunnableConfig): The configuration for the runnable.

        Returns:
            dict: The final state containing the updated messages.
        """
        while True:
            result = self.runnable.invoke(state)  # Invoke the LLM
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

# Graph
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
#memory = MemorySaver()
app = builder.compile()