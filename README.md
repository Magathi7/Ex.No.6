# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 25.09.2025
# Register no: 212223040108
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

#AI Tools Required:

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 

# Conclusion:
## Project Title: Multi-AI Comparison and Analysis Tool
## Aim

The aim of this project is to write and implement Python code that integrates with multiple AI tools (such as OpenAI GPT models, Cohere, or other AI APIs) to automate the process of:

Interacting with multiple AI APIs for the same prompt.

Comparing their outputs in terms of quality, clarity, and relevance.

Generating actionable insights based on the differences in outputs.

This can be useful for evaluating AI model performance, validating content, or automating decision-making.

## Required Tools and Libraries

requests – for interacting with REST APIs.

openai – for OpenAI GPT API.

Any other AI tool’s SDK (like cohere, huggingface_hub) depending on the APIs used.

pandas – for tabulating results.

difflib – for comparing text outputs.

## Algorithm / Workflow

Input Prompt: The user provides a text input or task.

Call Multiple AI APIs: Send the same prompt to each AI tool.

Collect Outputs: Store the responses from each AI model.

Compare Outputs: Analyze outputs for similarity, clarity, and coverage using text comparison tools or metrics.

Generate Insights: Summarize which AI tool performs best for a given task, or highlight differences.

Output Results: Print or save outputs and insights in a tabular format.

## Python Implementation
```
#Import necessary libraries
import openai
import requests
import difflib
import pandas as pd

# Configure API keys
openai.api_key = "YOUR_OPENAI_API_KEY"
COHERE_API_KEY = "YOUR_COHERE_API_KEY"

# Function to call OpenAI GPT API
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response['choices'][0]['message']['content'].strip()

# Function to call Cohere API
def get_cohere_response(prompt):
    url = "https://api.cohere.ai/generate"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "command-xlarge",
        "prompt": prompt,
        "max_tokens": 300
    }
    r = requests.post(url, headers=headers, json=data)
    if r.status_code == 200:
        return r.json()['generations'][0]['text'].strip()
    else:
        return f"Error: {r.status_code} - {r.text}"

# Function to compare outputs
def compare_texts(text1, text2):
    seq = difflib.SequenceMatcher(None, text1, text2)
    similarity = seq.ratio()
    return similarity

# Function to generate actionable insights
def generate_insights(responses):
    insights = []
    models = list(responses.keys())
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            sim = compare_texts(responses[models[i]], responses[models[j]])
            insights.append(f"Similarity between {models[i]} and {models[j]}: {sim:.2f}")
    
    return "\n".join(insights)

# Main workflow
if __name__ == "__main__":
    prompt = input("Enter your prompt for AI tools: ")

    responses = {}
    responses['OpenAI GPT'] = get_openai_response(prompt)
    responses['Cohere'] = get_cohere_response(prompt)
    
    # Display outputs
    print("\n=== AI Responses ===")
    for model, output in responses.items():
        print(f"\n[{model}]:\n{output}")

    # Generate comparison insights
    insights = generate_insights(responses)
    print("\n=== Comparison Insights ===")
    print(insights)

    # Save results to a table
    df = pd.DataFrame({
        'Model': list(responses.keys()),
        'Output': list(responses.values())
    })
    df.to_csv("ai_outputs_comparison.csv", index=False)
    print("\nResults saved to 'ai_outputs_comparison.csv'")
```

## Explanation

Input Prompt: The user provides a single task or question.

Multiple AI Tools Integration: The code calls OpenAI GPT and Cohere APIs for the same prompt. You can extend this to Hugging Face models or other AI tools.

Text Comparison: Uses Python’s difflib to calculate similarity between outputs. This is a simple way to quantify overlap and differences.

Insights Generation: Reports similarity scores and can highlight which model’s outputs differ significantly.

Result Storage: Saves outputs and analysis in a CSV file for record-keeping or further analysis.

## Future Enhancements

Integrate more AI tools for multi-model evaluation.

Use NLP-based metrics (like BLEU, ROUGE, or BERTScore) for deeper comparison instead of simple string similarity.

Include sentiment, readability, or factual correctness analysis.

Implement a GUI or dashboard for real-time comparison.

This script essentially automates prompt submission, output collection, comparison, and insight generation across multiple AI tools.


# Result: The corresponding Prompt is executed successfully.
