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

def agents():
    writer = autogen.AssistantAgent(
        name="writer",
        llm_config=config_list[0],
        system_message="""
        You are a professional writer, known for
        your insightful and engaging articles.
        You transform complex concepts into compelling narratives.
        Reply "TERMINATE" in the end when everything is done.
        """,
    )

    research_assistant = autogen.AssistantAgent(
        name="researcher",
        llm_config=config_list[0],
    )

    financial_assistant = autogen.AssistantAgent(
        name="financial",
        llm_config=config_list[0],
    )

    user = autogen.UserProxyAgent(
        name="user",
        human_input_mode="ALWAYS",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
    )

    return [user, writer, research_assistant, financial_assistant]

def run_model(prompt, tasks):
    user, writer, research_assistant, financial_assistant = agents()
    chat_results = user.initiate_chats(
        [
            {
                "recipient": financial_assistant,
                "message": prompt,
                "clear_history": True,
                "silent": False,
                "summary_method": "last_msg",
                "system_message": "End every message with 'TERMINATE'",
            },
            # {
            #     "recipient": research_assistant,
            #     "message": financial_tasks[1],
            #     "summary_method": "reflection_with_llm",
            # },
            # {
            #     "recipient": writer,
            #     "message": writing_tasks[0],
            #     "carryover": "I want to include a figure or a table of data in the blogpost.",
            # },
        ]
    )
    return chat_results


if __name__ == "__main__":
   run_model("Can you hear me?", ["can you hear me?"])