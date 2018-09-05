Header header

uint32 seq

#what kind of observations?
string source

#where was observations made?
#sensor frame to world frame transformation, stamped with time of localization reading
geometry_msgs/TransformStamped observation_transform

#list of words observed
int32[] words

#local position of each word in the sensor frame
float64[] word_pose

#local position of each word
int32[] word_scale

#words are in the range [vocabulary_begin, vocabulary_begin + vocabulary_size)
int32 vocabulary_begin
int32 vocabulary_size
