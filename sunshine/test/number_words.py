import rospy
from sunshine_msgs.msg import WordObservation, TopicMap
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
from sklearn.metrics import normalized_mutual_info_score


class NumberWordNode():
    def __init__(self, grid_size=10, vocab_size=100, num_topics=3, alpha=0.5, beta=0.1,
                 samples_per_topic=200, samples_per_document=1000, words_per_cell=10, neighborhood_size=1):
        self.grid_size = grid_size
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        assert (alpha > 0 and beta > 0)
        self.alphas = np.zeros((vocab_size,)) + alpha
        self.betas = np.zeros((num_topics,)) + beta
        self.seq = 0

        self.phi = np.ndarray((self.num_topics, self.vocab_size), dtype=np.float32)
        self.initialize_phi(n_samples=samples_per_topic)

        self.cell_poses = []
        self.theta = np.ndarray((grid_size ** 2, self.num_topics), dtype=np.float32)
        self.topic_labels = np.ndarray((grid_size ** 2,), dtype=np.int32)
        self.initialize_theta(samples_per_document, neighborhood_size)
        rospy.loginfo("GT topic labels: {}".format(self.topic_labels))

        words_topic = rospy.get_param('~words_topic', 'words')
        topics_topic = rospy.get_param('~topics_topic', 'topics')
        map_topic = rospy.get_param('~map_topic', 'topic_map')
        ref_topic = rospy.get_param('~ref_topic', '/topic_model/topic_map')
        rate = float(rospy.get_param('~rate', 1))

        words_pub = rospy.Publisher(words_topic, WordObservation)
        topics_pub = rospy.Publisher(topics_topic, WordObservation)
        word_msg = WordObservation()
        word_msg.observation_transform.transform.rotation.w = 1
        word_msg.vocabulary_begin = 0
        word_msg.vocabulary_size = vocab_size
        word_msg.source = 'number'

        for idx in range(len(self.cell_poses)):
            pose = self.cell_poses[idx]
            for i in range(words_per_cell):
                for j in range(3):
                    word_msg.word_pose.append(pose[j])
                word_msg.word_scale.append(1)

        topic_msg = deepcopy(word_msg)
        topic_msg.vocabulary_size = num_topics

        map_pub = rospy.Publisher(map_topic, TopicMap)
        map_msg = TopicMap()
        for pose in self.cell_poses:
            for j in range(3):
                map_msg.cell_poses.append(pose[j])
            map_msg.cell_ppx.append(0)
        map_msg.cell_topics = self.topic_labels.tolist()
        map_msg.observation_transform = word_msg.observation_transform
        map_msg.vocabulary_begin = 0
        map_msg.vocabulary_size = self.num_topics

        ref_sub = rospy.Subscriber(ref_topic, TopicMap, callback=lambda msg:
        rospy.loginfo('Mutual info: {}'.format(normalized_mutual_info_score(msg.cell_topics, map_msg.cell_topics))))

        while not rospy.is_shutdown():
            stamp = time() % 60
            word_msg.observation_transform.header.stamp.set(stamp, 0)
            word_msg.seq = self.seq
            topic_msg.observation_transform = word_msg.observation_transform
            topic_msg.seq = self.seq
            map_msg.observation_transform = word_msg.observation_transform
            map_msg.seq = self.seq

            word_msg.words = []
            topic_msg.words = []
            for idx in range(len(self.cell_poses)):
                words, topics = self.sample_cell_words(self.theta[idx, :], self.phi, words_per_cell)
                word_msg.words.extend(words)
                topic_msg.words.extend(topics)

            assert (len(word_msg.word_pose) == 3 * len(word_msg.words))
            assert (len(word_msg.word_pose) == len(topic_msg.word_pose))
            topics_pub.publish(topic_msg)
            words_pub.publish(word_msg)
            map_pub.publish(map_msg)
            rospy.sleep(1 / rate)
            self.seq += 1

    @staticmethod
    def draw_multinomial(weights):
        return np.random.multinomial(1, weights / float(np.sum(weights)))

    @staticmethod
    def sample_cell_weights(betas, n_samples):
        theta = np.copy(betas)
        for i in range(n_samples):
            theta += NumberWordNode.draw_multinomial(theta)
        return theta

    @staticmethod
    def sample_cell_words(theta, phi, n_words):
        words, topics = [], []
        for i in range(n_words):
            z = np.argmax(NumberWordNode.draw_multinomial(theta))
            w = np.argmax(NumberWordNode.draw_multinomial(phi[z, :]))
            words.append(w)
            topics.append(z)
        return words, topics

    def initialize_phi(self, n_samples, debug=True):
        for z in range(self.num_topics):
            self.phi[z, :] = np.copy(self.alphas)
            for i in range(n_samples):
                self.phi[z, :] += self.draw_multinomial(self.phi[z, :])
            if debug:
                bins = np.linspace(0, self.vocab_size, self.vocab_size)
                plt.bar(bins, self.phi[z, :], alpha=0.5, label=str(z), color=np.random.rand(3, 1))
        if debug:
            plt.show()

    def initialize_theta(self, n_samples, neighborhood_size, debug=False):
        cells = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                z = 0
                pose = (x, y, z)
                self.cell_poses.append(pose)
                cells[pose] = self.sample_cell_weights(self.betas, n_samples)
        idx = 0
        for pose in self.cell_poses:
            self.theta[idx, :] = cells[pose]
            for dx, dy in (zip([i for i in range(-neighborhood_size, neighborhood_size + 1) if i != 0], iter(int, 1))
                           + zip(iter(int, 1),
                                 [i for i in range(-neighborhood_size, neighborhood_size + 1) if i != 0])):
                new_pose = (pose[0] + dx, pose[1] + dy, pose[2])
                if new_pose in self.cell_poses:
                    self.theta[idx, :] += cells[new_pose]
            self.topic_labels[idx] = np.argmax(self.theta[idx, :])
            idx += 1
        if debug:
            print(self.theta)


rospy.init_node('number_words')
try:
    ne = NumberWordNode()
except rospy.ROSInterruptException:
    pass
