import re
import ast
from enum import IntEnum


class Labels(IntEnum):
	JoeRogan = 0
	learnprogramming = 1
	chemistry = 2
	NoStupidQuestions = 3
	Tinder = 4
	history = 5
	legaladvice = 6
	learnjava = 7
	Parenting = 8
	Home = 9
	technology = 10
	funny = 11
	Art = 12
	india = 13
	LifeProTips = 14
	drawing = 15
	AMA = 16
	techsupport = 17
	google = 18
	UTAustin = 19
	GetEmployed = 20
	dating_advice = 21
	computerscience = 22
	ethereum = 23
	Coronavirus = 24
	coding = 25
	CryptoCurrency = 26
	datascience = 27
	shrooms = 28
	explainlikeimfive = 29
	Fantasy = 30
	americanairlines = 31
	gaming = 32
	aww = 33
	gardening = 34
	aggies = 35
	AnimalCrossing = 36
	nasa = 37
	walmart = 38
	linux = 39
	travel = 40
	AskMen = 41
	Cooking = 42
	hiphopheads = 43
	videos = 44
	AskReddit = 45
	TAMU = 46
	btc = 47
	programming = 48
	texas = 49
	nfl = 50
	australia = 51
	iphone = 52
	memes = 53
	sports = 54
	philosophy = 55
	science = 56
	italy = 57
	ComputerEngineering = 58
	amazon = 59
	DIY = 60
	worldpolitics = 61
	atheism = 62
	MachineLearning = 63
	computer = 64
	help = 65
	hacking = 66
	worldnews = 67
	resumes = 68
	GetMotivated = 69
	mildlyinteresting = 70
	Music = 71
	JobFair = 72
	MovieDetails = 73
	biology = 74
	nba = 75
	manga = 76
	AskHistorians = 77
	nvidia = 78
	self = 79
	pharmacy = 80
	Showerthoughts = 81
	Economics = 82
	europe = 83
	lifehacks = 84
	keto = 85
	careerguidance = 86
	soccer = 87
	storage = 88
	marvelstudios = 89
	javahelp = 90
	cars = 91
	hiking = 92
	sysadmin = 93
	Bitcoin = 94
	trashy = 95
	financialindependence = 96
	AskWomen = 97
	UpliftingNews = 98
	Steam = 99
	csMajors = 100
	apple = 101
	CasualConversation = 102
	wallstreetbets = 103
	oculus = 104
	socialskills = 105
	gmaing = 106
	relationship_advice = 107
	movies = 108
	space = 109
	literature = 110
	tifu = 111
	camping = 112
	Minecraft = 113
	relationships = 114
	learnpython = 115
	wow = 116
	recipes = 117
	computervision = 118
	anime = 119
	snakes = 120
	conspiracy = 121
	news = 122
	cscareerquestions = 123
	modernwarfare = 124
	todayilearned = 125
	Astronomy = 126
	Jokes = 127
	exit = 128
	Futurology = 129
	teenagers = 130
	MMA = 131
	Python = 132
	food = 133
	guns = 134
	BasicIncome = 135
	unpopularopinion = 136
	dataisbeautiful = 137
	Documentaries = 138
	Fitness = 139
	jobs = 140
	pcgaming = 141
	askscience = 142
	datastorage = 143
	pics = 144
	bodybuilding = 145
	politics = 146
	offmychest = 147
	leagueoflegends = 148


class Dataset:
	TRAIN = "split_data/train.csv"
	# VALIDATE = "../data/validate.csv"
	TEST = "split_data/test.csv"

	def __init__(self, split='train'):
		self.data = []
		self.docs_path = self.TRAIN

		if split == 'test':
			self.docs_path = self.TEST
		elif split == 'val':
			self.docs_path = self.VALIDATE

		self.read_dataset()


	def fetch(self):
		return self.data


	def read_dataset(self):
		with open(self.docs_path, 'r') as file :
			next(file)
			for line in file:
				if line != '\n':
					split = line.strip().split(',')
					split[1] = split[1].replace('\'', '')
					# split[0] = re.sub(r'\r\n', " ", split[0])
					# split[0] = re.sub(r"[^a-zA-Z]+", " ", split[0])
					# split[0] = re.sub(r'[" "]+', " ", split[0])
					# print(split[1].strip())
					label = Labels.__dict__[split[1].strip()].value
					self.data.append((split[0], Labels(int(label))))
