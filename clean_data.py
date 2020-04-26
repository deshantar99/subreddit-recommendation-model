import pandas as pd
from sklearn import model_selection

subreddits = ['JoeRogan', 'learnprogramming', 'chemistry', 'NoStupidQuestions', 'Tinder', 'history', 'legaladvice', 'learnjava', 'Parenting', 'Home', 'technology', 'funny', 'Art', 'india', 'LifeProTips', 'drawing', 'ama', 'techsupport', 'google', 'UTAustin', 'GetEmployed', 'dating_advice', 'computerscience', 'ethereum', 'Coronavirus', 'coding', 'CryptoCurrency', 'datascience', 'shrooms', 'explainlikeimfive', 'Fantasy', 'americanairlines', 'gaming', 'aww', 'gardening', 'aggies', 'AnimalCrossing', 'nasa', 'walmart', 'linux', 'travel', 'AskMen', 'Cooking', 'hiphopheads', 'videos', 'AskReddit', 'TAMU', 'btc', 'programming', 'texas', 'nfl', 'australia', 'iphone', 'memes', 'sports', 'philosophy', 'science', 'italy', 'ComputerEngineering', 'amazon', 'DIY', 'worldpolitics', 'atheism', 'MachineLearning', 'computer', 'help', 'hacking', 'worldnews', 'resumes', 'GetMotivated', 'mildlyinteresting', 'Music', 'JobFair', 'MovieDetails', 'biology', 'nba', 'manga', 'AskHistorians', 'nvidia', 'self', 'pharmacy', 'Showerthoughts', 'Economics', 'europe', 'lifehacks', 'keto', 'careerguidance', 'soccer', 'storage', 'marvelstudios', 'javahelp', 'cars', 'hiking', 'sysadmin', 'Bitcoin', 'trashy', 'financialindependence', 'AskWomen', 'UpliftingNews', 'Steam', 'csMajors', 'apple', 'CasualConversation', 'wallstreetbets', 'oculus', 'socialskills', 'gmaing', 'relationship_advice', 'movies', 'space', 'literature', 'tifu', 'camping', 'Minecraft', 'relationships', 'learnpython', 'wow', 'recipes', 'computervision', 'anime', 'snakes', 'conspiracy', 'news', 'cscareerquestions', 'modernwarfare', 'todayilearned', 'Astronomy', 'Jokes', 'exit', 'Futurology', 'teenagers', 'MMA', 'Python', 'food', 'guns', 'BasicIncome', 'unpopularopinion', 'dataisbeautiful', 'Documentaries', 'Fitness', 'jobs', 'pcgaming', 'askscience', 'datastorage', 'pics', 'bodybuilding', 'politics', 'offmychest', 'leagueoflegends']


# subreddits = ['AskMen']

def convert_all_to_csv():
    for subreddit in subreddits:
        with open('reddit_data/redditData10_' + subreddit + '.txt', 'r') as txt_file, open('csv_files/' + subreddit + '.csv', 'w') as csv_file:
            csv_file.write('title,subreddit\n')
            for line in txt_file:
                if line != "\n":
                    line = line[1:-2]
                    csv_file.write(line.strip() + '\n')



def train_test_split():

    open('split_data/test.csv').close()
    open('split_data/train.csv').close()

    for subreddit in subreddits:
        with open('split_data/test.csv', 'a') as test_file, open('split_data/train.csv', 'a') as train_file:
            print("file: " + subreddit)
            Corpus = pd.read_csv('csv_files/' + subreddit + '.csv')
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['title'],Corpus['subreddit'],test_size=0.3)

            for i, title in Test_X.items():
                test_file.write(str(title) + ',' + str(Test_Y[i]) + '\n')

            for i, title in Train_X.items():
                train_file.write(str(title) + ',' + str(Train_Y[i]) + '\n')





convert_all_to_csv()
train_test_split()
