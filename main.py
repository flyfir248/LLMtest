from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Step 1: Choose a pre-trained model architecture
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Step 2: Install the required libraries and packages
# Already installed in the virtual environment

# Step 3: Load the pre-trained model
try:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except OSError:
    print(f"Model not found locally. Downloading {model_name}...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 4: Define a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


text="""
The World Chess Championship 2023 was a chess match between Ian Nepomniachtchi and Ding Liren to determine the new World Chess Champion. The match took place in Astana, Kazakhstan, from 9 April to 30 April 2023, and was a best of 14 games, plus tiebreaks.[1]

The previous champion Magnus Carlsen decided not to defend his title against Ian Nepomniachtchi, the winner of the Candidates Tournament 2022.[2][3] As a result, Nepomniachtchi played against Ding Liren, who finished second in the Candidates Tournament.
"""


question = "Who won the 2023 chess championship ?"

inputs = tokenizer.encode_plus(question, text, return_tensors="pt",max_length=512, truncation=True)

# Step 6: Run the input tensors through the pre-trained model
outputs = model(**inputs)

# Step 7: Retrieve the answer by decoding the output tensors
answer_start = outputs.start_logits.argmax(dim=-1).item()
answer_end = outputs.end_logits.argmax(dim=-1).item()
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1]))

print(f"Question: {question}")
print(f"Answer: {answer}")