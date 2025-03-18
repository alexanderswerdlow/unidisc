from transformers import AutoModelForSequenceClassification, AutoTokenizer
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
question, answer = "", ""
inputs = tokenizer(question, answer, return_tensors='pt')
score = rank_model(**inputs).logits[0].cpu().detach()
print(score)