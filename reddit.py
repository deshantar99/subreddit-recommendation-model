import praw
import pandas as pd
from praw.models import MoreComments
import re


arr = ['JoeRogan', 'learnprogramming', 'chemistry', 'NoStupidQuestions', 'Tinder', 'history', 'legaladvice', 'learnjava', 'Parenting', 'Home', 'technology', 'funny', 'Art', 'india', 'LifeProTips', 'drawing', 'ama', 'techsupport', 'google', 'UTAustin', 'GetEmployed', 'dating_advice', 'computerscience', 'ethereum', 'Coronavirus', 'coding', 'CryptoCurrency', 'datascience', 'shrooms', 'explainlikeimfive', 'Fantasy', 'americanairlines', 'gaming', 'aww', 'gardening', 'aggies', 'AnimalCrossing', 'nasa', 'walmart', 'linux', 'travel', 'AskMen', 'Cooking', 'hiphopheads', 'videos', 'AskReddit', 'TAMU', 'btc', 'programming', 'texas', 'nfl', 'australia', 'iphone', 'memes', 'sports', 'philosophy', 'science', 'italy', 'ComputerEngineering', 'amazon', 'DIY', 'worldpolitics', 'atheism', 'MachineLearning', 'computer', 'help', 'hacking', 'worldnews', 'resumes', 'GetMotivated', 'mildlyinteresting', 'Music', 'JobFair', 'MovieDetails', 'biology', 'nba', 'manga', 'AskHistorians', 'nvidia', 'self', 'pharmacy', 'Showerthoughts', 'Economics', 'europe', 'lifehacks', 'keto', 'careerguidance', 'soccer', 'storage', 'marvelstudios', 'javahelp', 'cars', 'hiking', 'sysadmin', 'Bitcoin', 'trashy', 'financialindependence', 'AskWomen', 'UpliftingNews', 'Steam', 'csMajors', 'apple', 'CasualConversation', 'wallstreetbets', 'oculus', 'socialskills', 'gmaing', 'relationship_advice', 'movies', 'space', 'literature', 'tifu', 'camping', 'Minecraft', 'relationships', 'learnpython', 'wow', 'recipes', 'computervision', 'anime', 'snakes', 'conspiracy', 'news', 'cscareerquestions', 'modernwarfare', 'todayilearned', 'Astronomy', 'Jokes', 'exit', 'Futurology', 'teenagers', 'MMA', 'Python', 'food', 'guns', 'BasicIncome', 'unpopularopinion', 'dataisbeautiful', 'Documentaries', 'Fitness', 'jobs', 'pcgaming', 'askscience', 'datastorage', 'pics', 'bodybuilding', 'politics', 'offmychest', 'leagueoflegends']


for sub in arr:
	f= open("reddit_data/redditData10_%s.txt" % sub,"w+")
	ml_subreddit = reddit.subreddit(sub)
	print(sub)
	for post in ml_subreddit.top(limit=1000):
		posts = []
		title = re.sub(r'\r\n', " ", post.title)
		title = re.sub(r"[^a-zA-Z]+", " ", title)
		title = re.sub(r'[" "]+', " ", title)
		posts.append([title.strip(), post.subreddit.display_name.strip()])
		f.writelines((str(posts[0]), "\n"))

	f.close()
