import rospy
from sunshine_msgs.msg import WordObservation
import matplotlib.pyplot as plt
import numpy as np
plt.interactive(False)

class WordAnalyzer():
    def __init__(self, grid_size=10, vocab_size=100):
        self.grid_size = grid_size
        self.vocab_size = vocab_size

        words_topic = rospy.get_param('~words_topic', '/rost_vision/words')
        rate = float(rospy.get_param('~rate', .1))

        words_sub = rospy.Subscriber(words_topic, WordObservation, self.on_words)

        while not rospy.is_shutdown():
            rospy.sleep(1 / rate)

    def on_words(self, words):
        """
        :type words: WordObservation
        """
        print('Received!')
        word_counts = {word: words.words.count(word) for word in set(words.words)}
        keys = sorted(list(word_counts.keys()))
        # print(len(words.words), len(keys))
        y_pos = np.arange(len(keys))
        word_counts = [word_counts[word] for word in keys]

        plt.bar(y_pos, word_counts, align='center', alpha=0.5)
        # plt.xticks(y_pos, keys)
        plt.ylabel('Count')
        plt.title('Word')
        print('Displaying!')
        plt.show()


rospy.init_node('number_words')
try:
    ne = WordAnalyzer()
except rospy.ROSInterruptException:
    pass
