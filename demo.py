from langchain_core.prompts import PromptTemplate

from AgentLLM import AgentLLM


def main() -> None:
    agent = AgentLLM("AIagent.env")

    prompt = PromptTemplate(
        template="你是一名算命大师，帮我起一个具有{country}国家特色，性别为{gender}的名字，姓氏为{firstname}。",
        input_variables=["country", "gender", "firstname"],
    )

    final_prompt = prompt.format(country="中国", gender="男", firstname="张")
    resp = agent.invoke(final_prompt)
    print(resp)


if __name__ == "__main__":
    main()
