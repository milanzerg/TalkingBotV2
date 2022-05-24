# This Python file uses the following encoding: utf-8
import telebot
from telebot import types

import nltk
import random

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from telega.project import config
from telega.project import BOT_DATA
bot = telebot.TeleBot(config.TOKEN)
admin_description = "Вас приветствует РАБ - Разумный Автономный Бот"


dataset = []  # [[example, intent], ...]

BOT_CONFIG = BOT_DATA.BOT_CONFIG

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        dataset.append([example, intent])

X_text = [x for x, y in dataset]
y = [y for x, y in dataset]

vectorizer = CountVectorizer(lowercase=True, ngram_range=(2, 3), analyzer='char_wb')
X = vectorizer.fit_transform(X_text)  # вектора примеров

tfidf_transformer = TfidfTransformer()
x_tf = tfidf_transformer.fit_transform(X)
X = x_tf

scores = []

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.35e-3, max_iter=1000, random_state=3)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

result = sum(scores) / 100
print(result)


def get_intent(text):
    vectors = vectorizer.transform([text])
    intent = clf.predict(vectors)[0]

    probas = clf.predict_proba(vectors)[0]
    index = list(clf.classes_).index(intent)
    proba = probas[index]

    if BOT_CONFIG['threshold'] <= proba:
        return intent


with open('dialogues/dialogues.txt', encoding='utf-8') as dialogues_file:
    content = dialogues_file.read()

dialogues = content.split('\n\n')
chit_chat_dataset = []  # [[question, answer], ...]

for dialogue in dialogues:
    replicas = dialogue.split('\n')
    replicas = [replica[2:].strip().lower() for replica in replicas]
    replicas = [replica for replica in replicas if replica]
    for i in range(len(replicas) - 1):
        chit_chat_dataset.append((replicas[i], replicas[i + 1]))

chit_chat_dataset = list(set(chit_chat_dataset))


def generate_random_answer(text):
    text = text.lower()

    for question, answer in chit_chat_dataset:
        if abs(len(text) - len(question)) / len(question) <= (1 - BOT_CONFIG['chit_chat_threshold']):
            distance = nltk.edit_distance(text, question)
            similarity = 1 - min(1, distance / len(question))
            if similarity >= BOT_CONFIG['chit_chat_threshold']:
                return answer


stats = {'rules': 0, 'generative': 0, 'fail': 0}


def generate_answer(text):
    intent = get_intent(text)

    if intent is not None:
        stats['rules'] += 1
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)

    random_answer = generate_random_answer(text)
    if random_answer is not None:
        stats['generative'] += 1
        return random_answer

    stats['fail'] += 1
    return random.choice(BOT_CONFIG['failure_phrases'])


@bot.message_handler(commands=['start'])
def welcome(message):
    sti = open('good.webp', 'rb')
    bot.send_sticker(message.chat.id, sti)

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Поболтать")

    markup.add(item1)

    bot.send_message(message.chat.id,
                     ("Хаюшки, {0.first_name}!\n" + admin_description).format(message.from_user, bot.get_me()),
                     parse_mode='html', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def starting(message):
    if message.chat.type == 'private':
        if message.text == 'Поболтать':
            bot.send_message(message.chat.id, "Что ж, давайте!")

        else:
            answer = generate_answer(message.text)
            print(answer)
            bot.send_message(message.chat.id, answer)
    print(stats)


bot.polling(none_stop=True)
