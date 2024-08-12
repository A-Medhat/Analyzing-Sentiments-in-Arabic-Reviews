import nltk
import pandas as pd
import torch
# from keras.utils.conv_utils import conv_input_length
from sklearn.feature_extraction.text import TfidfVectorizer
from tashaphyne.stemming import ArabicLightStemmer
import re
import emoji
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')

test = pd.read_csv("/kaggle/input/neural/test _no_label.csv")
df = pd.read_csv("/kaggle/input/neural/train1.csv")
# print(df)
df = df.iloc[:]
# print("print data", df)
# print(len(df))
# Remove duplicated
df.review_description.duplicated().sum()
df.drop(df[df.review_description.duplicated() == True].index, axis=0, inplace=True)

# Remove Punctuation
df.review_description = df.review_description.astype(str)
df.review_description = df.review_description.apply(
    lambda x: re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', x))
df.review_description = df.review_description.apply(lambda x: x.replace('؛', "", ))


# print("print data after pun", df.head())


# Define the function to remove consecutive duplicated Arabic words
def remove_duplicate_arabic_words(text):
    # Tokenize the text into words
    words = text.split()

    # Remove consecutive duplicated words
    unique_words = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i - 1]]

    # Join the unique words back into a sentence
    modified_text = ' '.join(unique_words)

    return modified_text


df['review_description'] = df['review_description'].apply(remove_duplicate_arabic_words)
# Remove StopWords
stopWords = list(set(stopwords.words("arabic")))  ## To remove duplictes and return to list again

# Some words needed to work with to will remove
for word in ['لا', 'لكن', 'ولكن']:
    stopWords.remove(word)
df.review_description = df.review_description.apply(
    lambda x: " ".join([word for word in x.split() if word not in stopWords]))
# print("print data after removing stop words", df.head())

# Replace Emoji by Text
emojis = {"🙂": "يبتسم", "😂": "يضحك", "💔": "قلب حزين", "🙂": "يبتسم", "❤": "حب", "❤": "حب", "😍": "حب", "😭": "يبكي",
          "😢": "حزن", "😔": "حزن", "♥": "حب", "💜": "حب", "😅": "يضحك", "🙁": "حزين", "💕": "حب", "💙": "حب", "😞": "حزين",
          "😊": "سعادة", "👏": "يصفق", "👌": "احسنت", "😴": "ينام", "😀": "يضحك", "😌": "حزين", "🌹": "وردة", "🙈": "حب",
          "😄": "يضحك", "😐": "محايد", "✌": "منتصر", "✨": "نجمه", "🤔": "تفكير", "😏": "يستهزء", "😒": "يستهزء", "🙄": "ملل",
          "😕": "عصبية", "😃": "يضحك", "🌸": "وردة", "😓": "حزن", "💞": "حب", "💗": "حب", "😑": "منزعج", "💭": "تفكير",
          "😎": "ثقة", "💛": "حب", "😩": "حزين", "💪": "عضلات", "👍": "موافق", "🙏🏻": "رجاء طلب", "😳": "مصدوم", "👏🏼": "تصفيق",
          "🎶": "موسيقي", "🌚": "صمت", "💚": "حب", "🙏": "رجاء طلب", "💘": "حب", "🍃": "سلام", "☺": "يضحك", "🐸": "ضفدع",
          "😶": "مصدوم", "✌": "مرح", "✋🏻": "توقف", "😉": "غمزة", "🌷": "حب", "🙃": "مبتسم", "😫": "حزين", "😨": "مصدوم",
          "🎼 ": "موسيقي", "🍁": "مرح", "🍂": "مرح", "💟": "حب", "😪": "حزن", "😆": "يضحك", "😣": "استياء", "☺": "حب",
          "😱": "كارثة", "😁": "يضحك", "😖": "استياء", "🏃🏼": "يجري", "😡": "غضب", "🚶": "يسير", "🤕": "مرض", "‼": "تعجب",
          "🕊": "طائر", "👌🏻": "احسنت", "❣": "حب", "🙊": "مصدوم", "💃": "سعادة مرح", "💃🏼": "سعادة مرح", "😜": "مرح",
          "👊": "ضربة", "😟": "استياء", "💖": "حب", "😥": "حزن", "🎻": "موسيقي", "✒": "يكتب", "🚶🏻": "يسير", "💎": "الماظ",
          "😷": "وباء مرض", "☝": "واحد", "🚬": "تدخين", "💐": "ورد", "🌞": "شمس", "👆": "الاول", "⚠": "تحذير",
          "🤗": "احتواء", "✖": "غلط", "📍": "مكان", "👸": "ملكه", "👑": "تاج", "✔": "صح", "💌": "قلب", "😲": "مندهش",
          "💦": "ماء", "🚫": "خطا", "👏🏻": "برافو", "🏊": "يسبح", "👍🏻": "تمام", "⭕": "دائره كبيره", "🎷": "ساكسفون",
          "👋": "تلويح باليد", "✌🏼": "علامه النصر", "🌝": "مبتسم", "➿": "عقده مزدوجه", "💪🏼": "قوي", "📩": "تواصل معي",
          "☕": "قهوه", "😧": "قلق و صدمة", "🗨": "رسالة", "❗": "تعجب", "🙆🏻": "اشاره موافقه", "👯": "اخوات", "©": "رمز",
          "👵🏽": "سيده عجوزه", "🐣": "كتكوت", "🙌": "تشجيع", "🙇": "شخص ينحني", "👐🏽": "ايدي مفتوحه", "👌🏽": "بالظبط",
          "⁉": "استنكار", "⚽": "كوره", "🕶": "حب", "🎈": "بالون", "🎀": "ورده", "💵": "فلوس", "😋": "جائع", "😛": "يغيظ",
          "😠": "غاضب", "✍🏻": "يكتب", "🌾": "ارز", "👣": "اثر قدمين", "❌": "رفض", "🍟": "طعام", "👬": "صداقة", "🐰": "ارنب",
          "☂": "مطر", "⚜": "مملكة فرنسا", "🐑": "خروف", "🗣": "صوت مرتفع", "👌🏼": "احسنت", "☘": "مرح", "😮": "صدمة",
          "😦": "قلق", "⭕": "الحق", "✏": "قلم", "ℹ": "معلومات", "🙍🏻": "رفض", "⚪": "نضارة نقاء", "🐤": "حزن", "💫": "مرح",
          "💝": "حب", "🍔": "طعام", "❤": "حب", "✈": "سفر", "🏃🏻‍♀": "يسير", "🍳": "ذكر", "🎤": "مايك غناء", "🎾": "كره",
          "🐔": "دجاجة", "🙋": "سؤال", "📮": "بحر", "💉": "دواء", "🙏🏼": "رجاء طلب", "💂🏿 ": "حارس", "🎬": "سينما",
          "♦": "مرح", "💡": "قكرة", "‼": "تعجب", "👼": "طفل", "🔑": "مفتاح", "♥": "حب", "🕋": "كعبة", "🐓": "دجاجة",
          "💩": "معترض", "👽": "فضائي", "☔": "مطر", "🍷": "عصير", "🌟": "نجمة", "☁": "سحب", "👃": "معترض", "🌺": "مرح",
          "🔪": "سكينة", "♨": "سخونية", "👊🏼": "ضرب", "✏": "قلم", "🚶🏾‍♀": "يسير", "👊": "ضربة", "◾": "وقف", "😚": "حب",
          "🔸": "مرح", "👎🏻": "لا يعجبني", "👊🏽": "ضربة", "😙": "حب", "🎥": "تصوير", "👉": "جذب انتباه", "👏🏽": "يصفق",
          "💪🏻": "عضلات", "🏴": "اسود", "🔥": "حريق", "😬": "عدم الراحة", "👊🏿": "يضرب", "🌿": "ورقه شجره", "✋🏼": "كف ايد",
          "👐": "ايدي مفتوحه", "☠": "وجه مرعب", "🎉": "يهنئ", "🔕": "صامت", "😿": "وجه حزين", "☹": "وجه يائس", "😘": "حب",
          "😰": "خوف و حزن", "🌼": "ورده", "💋": "بوسه", "👇": "لاسفل", "❣": "حب", "🎧": "سماعات", "📝": "يكتب", "😇": "دايخ",
          "😈": "رعب", "🏃": "يجري", "✌🏻": "علامه النصر", "🔫": "يضرب", "❗": "تعجب", "👎": "غير موافق", "🔐": "قفل",
          "👈": "لليمين", "™": "رمز", "🚶🏽": "يتمشي", "😯": "متفاجأ", "✊": "يد مغلقه", "😻": "اعجاب", "🙉": "قرد",
          "👧": "طفله صغيره", "🔴": "دائره حمراء", "💪🏽": "قوه", "💤": "ينام", "👀": "ينظر", "✍🏻": "يكتب", "❄": "تلج",
          "💀": "رعب", "😤": "وجه عابس", "🖋": "قلم", "🎩": "كاب", "☕": "قهوه", "😹": "ضحك", "💓": "حب", "☄ ": "نار",
          "👻": "رعب", "❎": "خطء", "🤮": "حزن", '🏻': "احمر"}
emoticons_to_emoji = {":)": "🙂", ":(": "🙁", "xD": "😆", ":=(": "😭", ":'(": "😢", ":'‑(": "😢", "XD": "😂", ":D": "🙂",
                      "♬": "موسيقي", "♡": "❤", "☻": "🙂"}


def checkemojie(text):
    emojistext = []
    for char in text:
        if any(emoji.distinct_emoji_list(char)) and char in emojis.keys():
            emojistext.append(emojis[emoji.distinct_emoji_list(char)[0]])
    return " ".join(emojistext)


def emojiTextTransform(text):
    cleantext = re.sub(r'[^\w\s]', '', text)
    return cleantext + " " + checkemojie(text)


# Apply checkemojie and emojiTextTransform
df['review_description'] = df['review_description'].apply(lambda x: emojiTextTransform(x))
# print("print data after changing the emoji to text", df['review_description'].head())

# Remove Numbers
df.review_description = df.review_description.apply(lambda x: ''.join([word for word in x if not word.isdigit()]))

# Apply Stemming
arabic_stemmer = ArabicLightStemmer()
# Apply stemming to the 'review_description' column
df['review_description'] = df['review_description'].apply(
    lambda x: " ".join([arabic_stemmer.light_stem(word) for word in x.split()]))

# print(len(df))

df.dropna(subset=['review_description'], inplace=True)
review_description = df['review_description']
y = df['rating']  # 1,0,-1
total_words = df['review_description'].dropna().str.split().apply(len).sum()
tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(review_description)
wordindex = tokenizer.word_index
# print(wordindex)
seq = tokenizer.texts_to_sequences(review_description)
seq_pad = pad_sequences(seq, maxlen=75, padding="post", truncating="post")
# print(seq_pad.shape)#(500, 75)


# Assuming the preprocessing up to the padding has been done as in the previous code and stored in `seq_pad` and `y`

# Convert the pad sequences and labels to torch Tensors
X = torch.from_numpy(seq_pad).long()  # Encoded and padded sequences
# y = torch.tensor(y).long()  # Labels
from torch.utils.data import TensorDataset, DataLoader
