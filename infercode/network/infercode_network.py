import tensorflow.compat.v1 as tf
import sys
import logging

class InferCodeModel():
    LOGGER = logging.getLogger('InferCodeModel')

    def __init__(self, num_types, num_tokens, num_subtrees, num_languages,
                num_conv: int=2, node_type_dim:int=50 , node_token_dim: int=50,
                conv_output_dim=50, include_token=1, batch_size=10, learning_rate: float=0.001):
     
        self.num_types = num_types
        self.num_tokens = num_tokens
        self.num_subtrees = num_subtrees
        self.num_languages = num_languages

        self.placeholders = {}
        self.weights = {}
    
        self.include_token = include_token
        self.num_conv = num_conv
        self.conv_output_dim = conv_output_dim
        self.node_type_dim = node_type_dim
        self.node_token_dim = node_token_dim

        # node_dim is equal to conv_output_dim and subtree_dim
        self.node_dim = conv_output_dim
        self.subtree_dim = conv_output_dim

        self.batch_size = batch_size

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

        self.init_net()
        self.feed_forward()

    def init_net(self):
        """Initialize parameters"""
        with tf.compat.v1.name_scope('inputs'):
            # nodes = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
           
            self.placeholders["node_type"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name='tree_node_type')
            self.placeholders["node_tokens"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='tree_node_tokens')
            self.placeholders["children_node_tokens"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None, None), name='children_tokens') # batch_size x max_num_nodes x max_children x max_sub_tokens
            
            self.placeholders["language_index"] = tf.constant(0)

            language_matrix_index = [x for x in range(self.num_languages)]
            self.weights["language_embeddings"] = tf.compat.v1.one_hot(language_matrix_index, len(language_matrix_index))

            self.weights["node_type_embeddings"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.num_types, self.node_type_dim]), name='node_type_embeddings')
            if self.include_token == 1:
                self.LOGGER.info("Including token weights..........")            
                self.weights["node_token_embeddings"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.num_tokens, self.node_token_dim]), name='node_token_embeddings')
            else:
                self.LOGGER.info("Excluding token weights..........")

            self.placeholders["children_index"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='children_index') # batch_size x max_num_nodes x max_children
            self.placeholders["children_node_type"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='children_node_type') # batch_size x max_num_nodes x max_children
            
            self.placeholders["labels"] = tf.compat.v1.placeholder(tf.float32, shape=(None, None), name="labels")
            self.placeholders["dropout_rate"] = tf.compat.v1.placeholder(tf.float32)
            # self.placeholders['is_training'] = tf.placeholder(tf.bool, name="is_training")
            
            self.weights["subtree_embeddings"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.num_subtrees, self.subtree_dim]), name='subtree_embeddings')
            self.weights["subtree_embeddings_bias"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.num_subtrees]), name='subtree_embeddings_bias')

            for i in range(self.num_conv):
                self.weights["w_t_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_t_' + str(i))
                self.weights["w_l_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_l_' + str(i))
                self.weights["w_r_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_r_' + str(i))
                self.weights["b_conv_" + str(i)] = tf.Variable(tf.zeros([self.conv_output_dim,]),name='b_conv_' + str(i))

            self.weights["w_attention"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, 1]), name="w_attention")
            
    def feed_forward(self):
        with tf.compat.v1.name_scope('network'):  
                 
            # shape = (batch_size, max_tree_size, node_type_dim)
            # Example with batch size = 12: shape = (12, 48, 30)
            self.parent_node_type_embeddings = self.compute_parent_node_types_tensor(self.placeholders["node_type"], self.weights["node_type_embeddings"])

            # shape = (batch_size, max_tree_size, max_children, node_type_dim)
            # Example with batch size = 12: shape = (12, 48, 8, 30)
            self.children_node_type_embeddings = self.compute_children_node_types_tensor(self.parent_node_type_embeddings, self.placeholders["children_index"], self.node_type_dim)



            if self.include_token == 1:
                self.LOGGER.info("Including token information..........")
                # shape = (batch_size, max_tree_size, node_token_dim)
                # Example with batch size = 12: shape = (12, 48, 50))
                self.parent_node_token_embeddings = self.compute_parent_node_tokens_tensor(self.placeholders["node_tokens"], self.weights["node_token_embeddings"])
                
                # shape = (batch_size, max_tree_size, max_children, node_token_dim)
                # Example with batch size = 12: shape = (12, 48, 7, 50)
                self.children_node_token_embeddings = self.compute_children_node_tokens_tensor(self.placeholders["children_node_tokens"], self.node_token_dim, self.weights["node_token_embeddings"])
               
                # Batch normalization for the inputs for regularization
                # self.parent_node_type_embeddings = tf.layers.batch_normalization(self.parent_node_type_embeddings, training=self.placeholders['is_training'])
                # self.parent_node_token_embeddings = tf.layers.batch_normalization(self.parent_node_token_embeddings, training=self.placeholders['is_training'])
                # self.children_node_types_tensor = tf.layers.batch_normalization(self.children_node_types_tensor, training=self.placeholders['is_training'])
                # self.children_node_tokens_tensor = tf.layers.batch_normalization(self.children_node_tokens_tensor, training=self.placeholders['is_training'])

                # shape = (batch_size, max_tree_size, (node_type_dim + node_token_dim))
                # Example with batch size = 12: shape = (12, 48, (30 + 50))) = (12, 48, 80)
                self.parent_node_embeddings = tf.concat([self.parent_node_type_embeddings, self.parent_node_token_embeddings], -1)
                # self.parent_node_embeddings = tf.compat.v1.layers.dense(self.parent_node_embeddings, units=self.node_dim, activation=tf.compat.v1.nn.leaky_relu)
                # shape = (batch_size, max_tree_size, max_children, (node_type_dim + node_token_dim))
                # Example with batch size = 12: shape = (12, 48, 6, (30 + 50))) = (12, 48, 6, 80)
                self.children_embeddings = tf.concat([self.children_node_type_embeddings, self.children_node_token_embeddings], -1)
                # self.children_embeddings = tf.compat.v1.layers.dense(self.children_embeddings, units=self.node_dim, activation=tf.compat.v1.nn.leaky_relu)


            else:
                self.LOGGER.info("Excluding token information..........")
                # Example with batch size = 12: shape = (12, 48, (30 + 50))) = (12, 48, 80)
                self.parent_node_embeddings = self.parent_node_type_embeddings
                self.children_embeddings = self.children_node_type_embeddings
            
            self.language_tensor = self.compute_language_tensors(self.placeholders["language_index"], self.weights["language_embeddings"])
            self.language_tensor_for_parent = tf.reshape(self.language_tensor, [1, 1, self.num_languages])
            self.language_tensor_for_parent = tf.tile(self.language_tensor_for_parent, [tf.shape(input=self.parent_node_embeddings)[0], tf.shape(input=self.parent_node_embeddings)[1], 1])
            self.parent_node_embeddings = tf.concat([self.parent_node_embeddings, self.language_tensor_for_parent], 2)
            self.parent_node_embeddings = tf.compat.v1.layers.dense(self.parent_node_embeddings, units=self.node_dim, activation=None, use_bias=False)

            self.language_tensor_for_children= tf.reshape(self.language_tensor, [1, 1, 1, self.num_languages])
            self.language_tensor_for_children = tf.tile(self.language_tensor_for_children, [tf.shape(input=self.children_embeddings)[0], tf.shape(input=self.children_embeddings)[1],  tf.shape(input=self.children_embeddings)[2], 1])
            self.children_embeddings = tf.concat([self.children_embeddings, self.language_tensor_for_children], -1)
            self.children_embeddings = tf.compat.v1.layers.dense(self.children_embeddings, units=self.node_dim, activation=None, use_bias=False)

            """Tree based Convolutional Layer"""
            # Example with batch size = 12 and num_conv = 8: shape = (12, 48, 128, 8)
            # Example with batch size = 1 and num_conv = 8: shape = (1, 48, 128, 8)
            self.conv_output = self.conv_layer(self.parent_node_embeddings, self.children_embeddings, self.placeholders["children_index"], self.num_conv, self.node_dim)

            # self.conv_output = tf.concat(self.conv_output, axis=-1)
            
            self.code_vector, self.attention_scores = self.aggregation_layer(self.conv_output, self.weights["w_attention"])
            # self.language_label_logits = tf.compat.v1.layers.dense(self.code_vector, units=self.num_languages, activation=tf.compat.v1.nn.leaky_relu)


            # self.language_label_loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=self.language_label_logits, labels=self.placeholders["language_index"])
            # self.language_label_loss = tf.reduce_mean(input_tensor=self.language_label_loss)
            # self.subtree_label_logits = tf.matmul(self.code_vector, self.weights["subtree_embeddings"], transpose_b=True)
            # self.loss = self.loss_layer(self.logits, self.placeholders["labels"])

            self.sampled_softmax_loss = tf.nn.sampled_softmax_loss(weights=self.weights["subtree_embeddings"], 
                                                              biases=self.weights["subtree_embeddings_bias"], 
                                                              labels=self.placeholders["labels"], 
                                                              inputs=self.code_vector, 
                                                              num_sampled=200, 
                                                              num_classes=self.num_subtrees)
            self.loss = tf.reduce_mean(input_tensor=self.sampled_softmax_loss)
            # self.loss = self.sampled_softmax_loss + 0.1*self.language_label_loss


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training_point = self.optimizer.minimize(self.loss)

    def aggregation_layer(self, nodes_representation, w_attention):
        # nodes_representation is (batch_size, max_graph_size, self.node_dim)
       
        with tf.compat.v1.name_scope("global_attention"):
            batch_size = tf.shape(input=nodes_representation)[0]
            max_tree_size = tf.shape(input=nodes_representation)[1]

            # (batch_size * max_graph_size, self.node_dim)
            flat_nodes_representation = tf.reshape(nodes_representation, [-1, self.node_dim])
            aggregated_vector = tf.matmul(flat_nodes_representation, w_attention)

            attention_score = tf.reshape(aggregated_vector, [-1, max_tree_size, 1])

            """A note here: softmax will distributed the weights to all of the nodes (sum of node weghts = 1),
            an interesting finding is that for some nodes, the attention score will be very very small, i.e e-12, 
            thus making parts of aggregated vector becomes near zero and affect on the learning (very slow nodes_representationergence
            - Better to use sigmoid"""

           
            attention_weights = tf.nn.softmax(attention_score, axis=1)
            
            # attention_weights = tf.nn.sigmoid(attention_score)

            # TODO: reduce_max vs reduce_sum vs reduce_mean
            # if aggregation_type == 1:
            #     print("Using tf.reduce_sum...........")
            weighted_average_nodes = tf.reduce_sum(input_tensor=tf.multiply(nodes_representation, attention_weights), axis=1)
            # if aggregation_type == 2:
                # print("Using tf.reduce_max...........")
                # weighted_average_nodes = tf.reduce_max(tf.multiply(nodes_representation, attention_weights), axis=1)
            # if aggregation_type == 3:
                # print("Using tf.reduce_mean...........")
                # weighted_average_nodes = tf.reduce_mean(tf.multiply(nodes_representation, attention_weights), axis=1)

            return weighted_average_nodes, attention_weights



    # def aggregation_layer(self, conv):
    #     # conv is (batch_size, max_tree_size, conv_output_dim)
    #     with tf.name_scope("global_attention"):
    #         batch_size = tf.shape(conv)[0]
    #         max_tree_size = tf.shape(conv)[1]

    #         contexts_sum = tf.reduce_sum(conv, axis=1)
    #         contexts_sum_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(max_tree_size, -1)))
          
    #         return contexts_sum_average


    def conv_node(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, layer):
        """Perform convolutions over every batch sample."""
        with tf.compat.v1.name_scope('conv_node'):
            w_t, w_l, w_r = self.weights["w_t_" + str(layer)], self.weights["w_l_" + str(layer)], self.weights["w_r_" + str(layer)]
            b_conv = self.weights["b_conv_" + str(layer)]
       
            return self.conv_step(parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv)

    def conv_layer(self, parent_node_embeddings, children_embeddings, children_indices, num_conv, node_dim):
        with tf.compat.v1.name_scope('conv_layer'):
            # nodes = [
            #     tf.expand_dims(self.conv_node(parent_node_embeddings, children_embeddings, children_indices, node_dim, layer),axis=-1)
            #     for layer in range(num_conv)
            # ] 
            # nodes = []

            for layer in range(num_conv):
                parent_node_embeddings = self.conv_node(parent_node_embeddings, children_embeddings, children_indices, node_dim, layer)
                children_embeddings = self.compute_children_node_types_tensor(parent_node_embeddings, children_indices, node_dim)
                # nodes.append(tf.expand_dims(parent_node_embeddings, axis=-1))
                # nodes = tf.expand_dims(parent_node_embeddings, axis=-1)
            return parent_node_embeddings 

    def conv_step(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv):
        """Convolve a batch of nodes and children.
        Lots of high dimensional tensors in this function. Intuitively it makes
        more sense if we did this work with while loops, but computationally this
        is more efficient. Don't try to wrap your head around all the tensor dot
        products, just follow the trail of dimensions.
        """
        with tf.compat.v1.name_scope('conv_step'):
            # nodes is shape (batch_size x max_tree_size x node_dim)
            # children is shape (batch_size x max_tree_size x max_children)

            with tf.compat.v1.name_scope('trees'):
              
                # add a 4th dimension to the parent nodes tensor
                # nodes is shape (batch_size x max_tree_size x 1 x node_dim)
                parent_node_embeddings = tf.expand_dims(parent_node_embeddings, axis=2)
                # tree_tensor is shape
                # (batch_size x max_tree_size x max_children + 1 x node_dim)
                tree_tensor = tf.concat([parent_node_embeddings, children_embeddings], axis=2, name='trees')

            with tf.compat.v1.name_scope('coefficients'):
                # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
                c_t = self.eta_t(children_indices)
                c_r = self.eta_r(children_indices, c_t)
                c_l = self.eta_l(children_indices, c_t, c_r)

                # concatenate the position coefficients into a tensor
                # (batch_size x max_tree_size x max_children + 1 x 3)
                coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

            with tf.compat.v1.name_scope('weights'):
                # stack weight matrices on top to make a weight tensor
                # (3, node_dim, conv_output_dim)
                weights = tf.stack([w_t, w_r, w_l], axis=0)

            with tf.compat.v1.name_scope('combine'):
                batch_size = tf.shape(input=children_indices)[0]
                max_tree_size = tf.shape(input=children_indices)[1]
                max_children = tf.shape(input=children_indices)[2]

                # reshape for matrix multiplication
                x = batch_size * max_tree_size
                y = max_children + 1
                result = tf.reshape(tree_tensor, (x, y, node_dim))
                coef = tf.reshape(coef, (x, y, 3))
                result = tf.matmul(result, coef, transpose_a=True)
                result = tf.reshape(result, (batch_size, max_tree_size, 3, node_dim))

                # output is (batch_size, max_tree_size, conv_output_dim)
                result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

                # output is (batch_size, max_tree_size, conv_output_dim)

                output = tf.compat.v1.nn.leaky_relu(result + b_conv)
                # output = tf.compat.v1.nn.swish(result + b_conv)
                # output = tf.layers.batch_normalization(output, training=self.placeholders['is_training'])
                output_drop_out = tf.nn.dropout(output, rate=self.placeholders["dropout_rate"])  # DROP-OUT here
                return output_drop_out

    def compute_children_node_types_tensor(self, parent_node_embeddings, children_indices, node_type_dim):
        """Build the children tensor from the input nodes and child lookup."""
    
        max_children = tf.shape(input=children_indices)[2]
        batch_size = tf.shape(input=parent_node_embeddings)[0]
        num_nodes = tf.shape(input=parent_node_embeddings)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, node_type_dim))
        # vector_lookup is (batch_size x num_nodes x node_dim)
        vector_lookup = tf.concat([zero_vecs, parent_node_embeddings[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children_indices = tf.expand_dims(children_indices, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children_indices = tf.concat([batch_indices, children_indices], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x node_type_dim)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children_indices)

    def compute_language_tensors(self, language_index, language_embeddings):
        language_tensors =  tf.nn.embedding_lookup(params=language_embeddings,ids=language_index)
        return language_tensors

    def compute_parent_node_types_tensor(self, parent_node_types_indices, node_type_embeddings):
        parent_node_types_tensor =  tf.nn.embedding_lookup(params=node_type_embeddings,ids=parent_node_types_indices)
        return parent_node_types_tensor
    
    def compute_parent_node_tokens_tensor(self, parent_node_tokens_indices, node_token_embeddings):
        parent_node_tokens_tensor = tf.nn.embedding_lookup(params=node_token_embeddings, ids=parent_node_tokens_indices)
        parent_node_tokens_tensor = tf.reduce_sum(input_tensor=parent_node_tokens_tensor, axis=2)
        return parent_node_tokens_tensor

    # def compute_children_node_types_tensor(self, children_node_types_indices):
    #     children_node_types_tensor =  tf.nn.embedding_lookup(self.node_type_embeddings, children_node_types_indices)
    #     return children_node_types_tensor
    
    def compute_children_node_tokens_tensor(self, children_node_tokens_indices, node_token_dim, node_token_embeddings):
        batch_size = tf.shape(input=children_node_tokens_indices)[0]
        zero_vecs = tf.zeros((1, node_token_dim))
        vector_lookup = tf.concat([zero_vecs, node_token_embeddings[1:, :]], axis=0)
        children_node_tokens_tensor = tf.nn.embedding_lookup(params=vector_lookup, ids=children_node_tokens_indices)
        children_node_tokens_tensor = tf.reduce_sum(input_tensor=children_node_tokens_tensor, axis=3)
        return children_node_tokens_tensor

    def eta_t(self, children):
        """Compute weight matrix for how much each vector belongs to the 'top'"""
        with tf.compat.v1.name_scope('coef_t'):
            # children is shape (batch_size x max_tree_size x max_children)
            batch_size = tf.shape(input=children)[0]
            max_tree_size = tf.shape(input=children)[1]
            max_children = tf.shape(input=children)[2]
            # eta_t is shape (batch_size x max_tree_size x max_children + 1)
            return tf.tile(tf.expand_dims(tf.concat(
                [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
                axis=1), axis=0,
            ), [batch_size, 1, 1], name='coef_t')

    def eta_r(self, children, t_coef):
        """Compute weight matrix for how much each vector belogs to the 'right'"""
        with tf.compat.v1.name_scope('coef_r'):
            # children is shape (batch_size x max_tree_size x max_children)
            children = tf.cast(children, tf.float32)
            batch_size = tf.shape(input=children)[0]
            max_tree_size = tf.shape(input=children)[1]
            max_children = tf.shape(input=children)[2]

            # num_siblings is shape (batch_size x max_tree_size x 1)
            num_siblings = tf.cast(
                tf.math.count_nonzero(children, axis=2, keepdims=True),
                dtype=tf.float32
            )
            # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
            num_siblings = tf.tile(
                num_siblings, [1, 1, max_children + 1], name='num_siblings'
            )
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.minimum(children, tf.ones(tf.shape(input=children)))],
                axis=2, name='mask'
            )

            # child indices for every tree (batch_size x max_tree_size x max_children + 1)
            child_indices = tf.multiply(tf.tile(
                tf.expand_dims(
                    tf.expand_dims(
                        tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                        axis=0
                    ),
                    axis=0
                ),
                [batch_size, max_tree_size, 1]
            ), mask, name='child_indices')

            # weights for every tree node in the case that num_siblings = 0
            # shape is (batch_size x max_tree_size x max_children + 1)
            singles = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.fill((batch_size, max_tree_size, 1), 0.5),
                 tf.zeros((batch_size, max_tree_size, max_children - 1))],
                axis=2, name='singles')

            # eta_r is shape (batch_size x max_tree_size x max_children + 1)
            return tf.compat.v1.where(
                tf.equal(num_siblings, 1.0),
                # avoid division by 0 when num_siblings == 1
                singles,
                # the normal case where num_siblings != 1
                tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
                name='coef_r'
            )

    def eta_l(self, children, coef_t, coef_r):
        """Compute weight matrix for how much each vector belongs to the 'left'"""
        with tf.compat.v1.name_scope('coef_l'):
            children = tf.cast(children, tf.float32)
            batch_size = tf.shape(input=children)[0]
            max_tree_size = tf.shape(input=children)[1]
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                    tf.minimum(children, tf.ones(tf.shape(input=children)))],
                axis=2,
                name='mask'
            )

            # eta_l is shape (batch_size x max_tree_size x max_children + 1)
            return tf.multiply(
                tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
            )

    def loss_layer(self, logits_node, labels):
        """Create a loss layer for training."""
    
        with tf.compat.v1.name_scope('loss_layer'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits_node, name='cross_entropy'
            )

            loss = tf.reduce_mean(input_tensor=cross_entropy, name='cross_entropy_mean')
            return loss
 
    # def loss_layer(self, training_logits, targets, target_mask):
    #     """Create a loss layer for training."""
       
    #     with tf.name_scope('loss_layer'):
    #         loss = tf.contrib.seq2seq.sequence_loss(training_logits,targets,target_mask)

    #         return loss

    # def out_layer(self, logits_node):
    #     """Apply softmax to the output layer."""
    #     with tf.name_scope('output'):
    #         return tf.nn.softmax(logits_node)
