from googletrans import Translator
translator = Translator()

text = "امروز در وقت تنفس مجلس برای نماز و ناهار دکتر عارف که در این مدت دائما در مجلس حضور داشت از این وقت یک ساعت و اندی بخشی را برای پیگیر امورات دانشجویانش در دانشگاه شریف از طریق فضای مجازی اختصاص داد. همیشه مسائل علمی و نهاد علم برای ایشان در اولویت بود.:"
result = translator.translate(text, dest='en')
print(result.text)
print(result.src)
print(result.dest)
