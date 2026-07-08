"""Run this model in Python

> pip install openai
"""
import os
from openai import OpenAI

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings. 
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"],
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Act as a blogger and Explain the user topic with 5 bullet points. Summerize the response to under 1000 words. Output should be in json format with each bullet point as list of bullet keywork. title, header and conclusion to be used as keywords for blog title, introduction content, and conclusion respectively. ",
        },
        {
            "role": "user",
            "content": "topic: microservices",
        }
    ],
    model="openai/gpt-4o",
    max_tokens=1024,
    top_p=1
)

print(response.choices[0].message.content)

# Response received - 
"""
{
  "title": "Understanding Microservices: A Comprehensive Guide",
  "header": "Introduction to Microservices Architecture",
  "content": [
    {
      "bullet": "Decoupled Architecture",
      "description": "Microservices involve breaking down an application into smaller, independent components that can be developed, deployed, and maintained separately."
    },
    {
      "bullet": "Scalability and Flexibility",
      "description": "Each microservice can be scaled individually, allowing businesses to adapt to growth or high-demand areas without affecting the entire system."
    },
    {
      "bullet": "Technology Diversity",
      "description": "Microservices enable the use of different programming languages, databases, or tools for each service, fostering innovation and addressing specific service needs."
    },
    {
      "bullet": "Resilience and Fault Isolation",
      "description": "Failures in one microservice have limited impact on the rest of the system, ensuring better fault tolerance and reliability overall."
    },
    {
      "bullet": "Continuous Deployment and Agility",
      "description": "Microservices promote faster delivery cycles and agile evolution of software, allowing teams to roll out changes or features independently."
    }
  ],
  "conclusion": "Microservices empower businesses by enabling modular application development, improving scalability, resilience, and fostering innovation with diverse technology choices."
}
"""
