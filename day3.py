from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

def main():
    system_template = ("You are a judge trying to give out a score: [\"Needs Improvement\", \"Passing\", \"Outstanding\"]. "
                       "You are judging based on how funny it is."
                       "Give a score and a reason why"
                       )
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        ('user', '{text}')
    ])

    model = OllamaLLM(model='gemma2:2b')

    parser = StrOutputParser()

    to_text_chain = prompt_template | model | parser
    text = to_text_chain.invoke({'text': 'hello world'})
    # equivalent statment below
    # chain = lambda x: parser.invoke(model.invoke(prompt_template.invoke(x)))
    # text2 = parser.invoke(model.invoke(prompt_template.invoke({'text': 'hello world'})))
    print(text)
    # print(text2)

if __name__ == '__main__':
    main()

