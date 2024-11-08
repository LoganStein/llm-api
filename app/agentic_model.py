import os
from dotenv import load_dotenv
import autogen

load_dotenv()

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
]
llm_config = {"config_list": config_list, "cache_seed": 32}


def agents():
    planner_agent = autogen.AssistantAgent(
        name="planner",
        llm_config=llm_config,
        system_message="""
        You are an AI software architect agent. Your task is to take the user's prompt and come up with a detailed plan for an expert programmer to implement to fulfill the request. Once you have created the step by step plan, you will pass it to the coder agent who will write the code to carry out the tasks. 
        Ensure the plan is clear, actionable, and broken down into manageable steps. 
        Do not try to fulfil the user's request directly. Your only job is to create the plan to fulfil the request. 
        """,
    )

    executor_agent = autogen.AssistantAgent(
        name="executor",
        llm_config=llm_config,
        system_message="""
        You are an AI assistant. Your task is to go through the plan you recieve from the planner agent and execute the steps in the plan.
        Do not write and code. Your job is to follow the plan and provide the user with the results of each step.
        Reply "TERMINATE" in the end when you have completed all the steps in the plan.
        """
    )

    coder_agent = autogen.AssistantAgent(
        name="coder",
        llm_config=llm_config,
        system_message="""
        You are an expert software engineer. You should write code that fulfils the user's request and follows the plan provided by the planner. The code that you write should never require an API key. Only use free and open source data sources.
        """

    )

    user = autogen.UserProxyAgent(
        name="user",
        human_input_mode="TERMINATE",
        system_message="A human admin.",
        # is_termination_msg=lambda x: x.get("content", "") and x.get(
        #     "content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "last_n_messages": 2,
            "work_dir": "groupchat",
            "use_docker": False,
        },
    )

    return [user, planner_agent, executor_agent, coder_agent]


def run_model(prompt):
    user, planner_agent, executor_agent, coder_agent = agents()
    groupchat = autogen.GroupChat(
        agents=[user, planner_agent, coder_agent], messages=[], max_round=10)
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config)
    chat_results = user.initiate_chat(manager, message=prompt, summary_method="reflection_with_llm",)
    return chat_results


if __name__ == "__main__":
    run_model(
        "Get the current stock price of NTRS.")
