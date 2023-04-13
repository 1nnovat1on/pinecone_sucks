import os
import re
from collections import defaultdict
from transformers import pipeline

# stores longterm memory
FOLDER_NAME = "long_term_memory"

def get_relevant_files(query = None):
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    def build_index(folder_path):
        index = defaultdict(list)
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                words = tokenize(filename[:-4])  # Remove .txt extension before tokenizing
                for word in words:
                    index[word].append(filename)
        return index

    def search(query, folder_path):
        index = build_index(folder_path)
        query_words = tokenize(query)
        
        relevance_scores = defaultdict(int)
        for word in query_words:
            if word in index:
                for filename in index[word]:
                    relevance_scores[filename] += 1
                    
        relevant_files = sorted(relevance_scores, key=relevance_scores.get, reverse=True)

        results = []
        for filename in relevant_files:
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                results.append({"filename": filename, "content": content})

        return results

    
    folder_path = os.path.join("C:\\", FOLDER_NAME)  # Specify the root of the C: drive

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    results = search(query, folder_path)

    # Set up the Transformers pipeline
    nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

    formatted_results = []
    for result in results:
        context = result['content']
        answer = nlp(question=query, context=context)
        formatted_results.append(f"Filename: {result['filename']}\nAnswer: {answer['answer']}\n")

    final_result = "\n".join(formatted_results)

    print(final_result)

    return final_result

# Feed final_result to the LLM

