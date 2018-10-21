import re
import os
import urllib.request
import json
import praw
import requests

pattern_album = re.compile(r"(?:i|w{3})?\.?imgur.com/a/[a-zA-Z0-9#]+$")
pattern_imagelink = re.compile(r"(?:i|w{3})?\.?imgur.com/[a-zA-Z0-9]+$")
pattern_image = re.compile(r"\.\w+$")

# image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
# valid_filters = ["day", "hour", "week", "month", "year", "all"]

# pattern_album_scrapper = re.compile('<meta property="og:image" content="http://((?:i|w{3})?\.?imgur.com/\w+\.\w+)" />')

# def direct_image_downloader(i, data, target):
#     target_folder = os.path.join(target, "%d_"%i + data["author"])
#     if not os.path.exists(target_folder):
#         os.mkdir(target_folder)
#     extension = data["url"].split(".")[-1]
#     final_target = os.path.join(target_folder, data["author"])+".%s"%extension
#     if not os.path.exists(final_target):
#         img = urllib.request.urlopen(data["url"])
#         t_image = open(final_target, "wb")
#         t_img.write(img.read())

# sort_filter = 'top'
# n = 5
# top_filter = 'all'
# subreddit = 'me_irl'

# target = os.path.join(os.getcwd(),subreddit + "_dump")

# response = urllib.request.urlopen('http://www.reddit.com/r/' + subreddit + '/top/.json?sort=top&t=' + top_filter + '&limit=' + str(n))

# data = json.load(response)
# for i, element in enumerate(data["data"]["children"]):
#     flag = False
#     if pattern_image.search(element["data"]["url"]):
#         direct_image_downloader(i, element["data"], target)
#         flag = True
#     if flag:
#         print("Done.")

reddit = praw.Reddit(client_id='aIe9JHEIL2HJYw', client_secret = 'AtnS3YR1meKQpKweTk16viCsTHk', user_agent='ChickenChopSuey')
subreddit = 'me_irl'
submissions = reddit.subreddit(subreddit).top(limit=2)
target = os.path.join(os.getcwd(),'reddit', subreddit)

if not os.path.exists('reddit/{}'.format(subreddit)):
    os.mkdir('reddit/{}'.format(subreddit))

for i, post in enumerate(submissions):
    url = (post.url)
    file_name = 'meme_%d.jpg' % i
    print(file_name)

    req = requests.get(url)
    with open('reddit/' + subreddit + '/' + file_name, "wb") as f:
        f.write(req.content)

