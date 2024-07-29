from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "facebook/blenderbot-400M-distill"

# Charger le modèle et le tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Suivi de l'historique de conversation
conversation_history = []

input_text = "hello, how are you doing?"

# Ajouter l'entrée actuelle à l'historique
conversation_history.append(input_text)

# Créer la chaîne d'historique
history_string = "\n".join(conversation_history)

# Encoder les inputs
inputs = tokenizer.encode_plus(history_string, return_tensors="pt")

# Générer une réponse
outputs = model.generate(**inputs)

# Décoder la réponse
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Afficher les résultats
print("Inputs:", inputs)
print("Response:", response)

# Ajouter la réponse à l'historique
conversation_history.append(response)
