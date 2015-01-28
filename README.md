Tweeting using a Markov chain trained on the tweets from [@parrotization](https://twitter.com/parrotization)'s followers.

Usage
-----

1. First, make a new Twitter app and then run `python parrotization.py setup` and enter the required info.
2. Then, once you have some followers, run `python parrotization.py update` to download a bunch of tweets and train the model.
3. To queue up some tweets, run `python parrotization.py queue` and curate the shit.
4. Finally, run `python parrotization.py tweet` to post a queued tweet.

Author & License
----------------

Dan Foreman-Mackey & MIT
