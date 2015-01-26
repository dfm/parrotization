#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["setup", "update_list", "update_database", "write_tweet"]

import os
import json
import nltk
import tweepy
import string
import numpy as np
import cPickle as pickle
from collections import defaultdict

PROJECTNAME = "parrotization"
DATABASE_FILE = "{0}.pkl".format(PROJECTNAME)
SETTINGS_FILE = "{0}.json".format(PROJECTNAME)
START = "<S>"
STOP = "</S>"


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
    else:
        settings = {}
    return settings


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_api():
    settings = load_settings()
    auth = tweepy.OAuthHandler(settings["consumer_key"],
                               settings["consumer_secret"])
    auth.secure = True
    auth.set_access_token(settings["user_key"], settings["user_secret"])
    return tweepy.API(auth,
                      wait_on_rate_limit=True,
                      wait_on_rate_limit_notify=True)


def _default():
    return defaultdict(int)


def load_db():
    if not os.path.exists(DATABASE_FILE):
        bigrams = defaultdict(_default)
        trigrams = defaultdict(_default)
        return (bigrams, trigrams)
    with open(DATABASE_FILE, "r") as f:
        return pickle.load(f)


def save_db(db):
    with open(DATABASE_FILE, "wb") as f:
        return pickle.dump(db, f, -1)


def setup(clobber=False):
    settings = load_settings()

    # Get the app information.
    if (clobber or "consumer_key" not in settings or
            "consumer_secret" not in settings):
        print("Enter some info about your app")
        settings["consumer_key"] = raw_input("Consumer key: ")
        settings["consumer_secret"] = raw_input("Consumer secret: ")

    # Authorize the user.
    if clobber or "user_key" not in settings or "user_secret" not in settings:
        auth = tweepy.OAuthHandler(settings["consumer_key"],
                                   settings["consumer_secret"],
                                   "oob")
        url = auth.get_authorization_url()
        print("Go to this URL:\n{0}".format(url))
        pin = raw_input("Enter the PIN: ")
        auth.get_access_token(pin)

        settings["user_key"] = auth.access_token.key
        settings["user_secret"] = auth.access_token.secret
    save_settings(settings)


def update_list():
    # Get the initial settings.
    api = get_api()
    settings = load_settings()
    if "list_slug" not in settings:
        settings["list_slug"] = api.create_list("cast").slug
        save_settings(settings)
    if "screen_name" not in settings:
        settings["screen_name"] = api.me().screen_name
        save_settings(settings)

    # Add all the followers to the list.
    owner, list_slug = settings["screen_name"], settings["list_slug"]
    api.add_list_members(user_id=api.followers_ids(),
                         owner_screen_name=owner, slug=list_slug)


def update_database():
    # Get all of the recent tweets in the timeline.
    api = get_api()
    settings = load_settings()
    bigrams, trigrams = load_db()
    owner, list_slug = api.me().screen_name, settings["list_slug"]
    for tweet in tweepy.Cursor(api.list_timeline, owner_screen_name=owner,
                               since_id=settings.get("since_id", None),
                               include_rts=False,
                               slug=list_slug).items(1000):
        # Tokenize the tweet.
        text = tweet.text
        a, b = "://", "URLURLURL"
        text = text.replace(a, b)
        tokens = [w.replace(b, a) for w in nltk.word_tokenize(text)]
        tokens = [START, START]+tokens+[STOP, STOP]

        # Update the id of the most recently seen tweet.
        settings["since_id"] = max(tweet.id, settings.get("since_id", 0))

        # Update the bigram and trigram dictionaries.
        for i in range(2, len(tokens)):
            bigrams[tokens[i-1]][tokens[i]] += 1
            trigrams[tokens[i-2]+" "+tokens[i-1]][tokens[i]] += 1

    # Save the database and the settings file.
    save_db((bigrams, trigrams))
    save_settings(settings)


def build_tweet(words, api, settings):
    s = " "
    for i, w in enumerate(words):
        if i > 0 and words[i-1] == "@":
            try:
                f, _ = api.show_friendship(
                    source_screen_name=settings["screen_name"],
                    target_screen_name=w)
            except tweepy.error.TweepError:
                is_follower = False
            else:
                is_follower = f.followed_by
            if is_follower:
                s += w + " "
            else:
                s = s[:-1] + "." + w + " "
        elif w.startswith("'") or w in ["n't"]:
            s = s[:-1] + w + " "
        elif not len(w.strip(string.punctuation)):
            if w in ["(", "{", "@", "#", "&", "``"]:
                s += w
            else:
                s = s[:-1] + w + " "
        else:
            s += w + " "
    s = s.strip()

    # Finally match any missing parens.
    if "(" in s and ")" not in s:
        s += ")"
    if ")" in s and "(" not in s:
        s = "(" + s

    s = s.replace("``", "\"").replace("''", "\"")

    return s


def write_tweet(alpha=0.6):
    api = get_api()
    settings = load_settings()
    bigrams, trigrams = load_db()

    tweet = [START, START]
    while True:
        b_prob = bigrams[tweet[-1]]
        t_prob = trigrams[tweet[-2]+" "+tweet[-1]]

        b_norm = sum(b_prob.values())
        t_norm = sum(t_prob.values())
        if b_norm < 1 or t_norm < 1:
            continue

        words, probs = [], []
        for w in set(b_prob.keys()) | set(t_prob.keys()):
            words.append(w)
            probs.append(alpha * t_prob.get(w, 0.0)/t_norm
                         + (1-alpha) * b_prob.get(w, 0.0)/b_norm)

        word = np.random.choice(words, p=probs)
        if word == STOP:
            if len(tweet) > 6:
                break
            # Too short.
            tweet = [START, START]
            continue
        tweet.append(word)
        sent = build_tweet(tweet[2:], api, settings)
        if len(sent) > 140:
            # Too long.
            tweet = [START, START]
    return sent


if __name__ == "__main__":
    import sys

    if "setup" in sys.argv:
        setup()

    elif "update" in sys.argv:
        update_list()
        update_database()

    elif "print" in sys.argv:
        print(write_tweet())

    elif "tweet" in sys.argv:
        print(write_tweet())
