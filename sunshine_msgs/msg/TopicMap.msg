Header header

uint32 seq

#what kind of observations?
string source

#type of perplexity calculated
string ppx_type

#where was observations made?
#sensor frame to world frame transformation, stamped with time of localization reading
geometry_msgs/TransformStamped observation_transform

#list of topics observed
int32[] cell_topics

#local position of each topic in the global frame
float64[] cell_poses

#perplexity of each cell
float64[] cell_ppx

#topics are in the range [vocabulary_begin, vocabulary_begin + vocabulary_size)
int32 vocabulary_begin
int32 vocabulary_size
