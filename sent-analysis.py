from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# bu kısma veri girişi yapılıyor
tweet = 'I am very about my graduation is late because of disaster'

# metin ön işleme aşaması
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

# Doğal dil modeli ekleme
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# tensor aşaması
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

# Dil Modeli ve Tensorflowdan gelen çıktıları basma aşaması
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)
