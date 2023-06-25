import asyncio
from typing import List

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import config


async def get_company_name(llm: OpenAI, product: str) -> str:
    prompt = PromptTemplate.from_template(
        "What would be a good company name for a company that makes {product}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return {
        'Product': product,
        'Suggested Company Name': await chain.arun(product),
    }


async def get_company_names(products: List[str]):
    llm = OpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.9,
    )
    tasks = [get_company_name(llm, product) for product in products]
    res = await asyncio.gather(*tasks)
    return res


async def main():
    products = [
        'colorful socks',
        'financial analytics software for institutional investors',
        'cat toys',
        'novelty towels',
    ]
    print('Generating some creative company names for companies that produce the following products:')
    print(products)
    res = await get_company_names(products)
    [print(r) for r in res]


asyncio.run(main())
