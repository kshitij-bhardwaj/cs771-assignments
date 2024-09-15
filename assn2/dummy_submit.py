import numpy as np
import random

def my_fit( words, verbose = False ):
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	dt.fit( words, verbose )
	return dt

def my_predict( dt, bg ):
	return dt.predict( bg )

class Tree:
	def __init__( self, min_leaf_size = 1, max_depth = 1 ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words, verbose = False ):
		self.words = words
		self.root = Node( depth = 0, parent = None )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		# The root is trained with all the words
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )
	
	def predict( self, bg ):
		node = self.root
		
		# Go to left child or right child depending on selection criterion
		while not node.is_leaf:
			node = node.get_child( node.get_query in bg )
			
		# Dummy leaf action -- return the list of words
		# However, be careful that only 5 guesses are considered
		return [ self.words[i] for i in node.my_words_idx ]


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query = None
		self.history = []
	
	# Each node implements a get_query method that generates the
	# query that gets asked when we reach that node.
	def get_query( self ):
		return self.query
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbo should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	# Dummy leaf action -- return the list of words
	# However, be careful that only 5 guesses are considered
	def process_leaf( self, all_words, my_words_idx, history, verbose ):
		self.my_words_idx = my_words_idx 
	
	# Get the deduplicated, sorted and truncated list of bigrams for a word
	def get_bigrams( word, lim = 5 ):
		# Get all bigrams
		bg = map( ''.join, list( zip( word, word[1:] ) ) )
		# Remove duplicates and sort them
		bg = sorted( set( bg ) )
		# Make them into an immutable tuple and retain only the first few
		return tuple( bg )[:lim]
	
	def get_random_bigram():
		return chr( ord('a') + random.randint(0,25) ) + chr( ord('a') + random.randint(0,25) )
	
	# Dummy node splitting action -- use a random bigram as the query
	def process_node( self, all_words, my_words_idx, history, verbose ):
		# Generating a random bigram
		query = get_random_bigram()
		
		if ( query in history ) and verbose:
			print( f"Warning: bigram being re-used -- bigram {query} has already been used by an ancestor node as the splitting criterion. This is suboptimal!" )
		
		# Left child will contain words that have the chosen bigram
		# Right child will contain words that do not have the chosen bigram
		split_dict = {}
		split_dict[ True ] = []
		split_dict[ False ] = []
		
		for idx in my_words_idx:
			bg_list = get_bigrams( all_words[ idx ] )
			split[ query in bg_list ].append( idx )
			
		if len( split_dict.items() ) < 2 and verbose:
			print( "Warning: did not make any meaningful split with this query!" )
		
		return ( query, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.process_leaf( self.all_words, self.my_words_idx, self.history, verbose )
			if verbose:
				print( '█' )
		else:
			self.is_leaf = False
			( self.query, split_dict ) = self.process_node( self.all_words, self.my_words_idx, self.history, verbose )
			
			if verbose:
				print( self.query )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				if verbose:
					if i == len( split_dict ) - 1:
						print( fmt_str + "└───", end = '' )
						fmt_str += "    "
					else:
						print( fmt_str + "├───", end = '' )
						fmt_str += "│   "
				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				history = self.history.copy()
				history.append( self.query )
				self.children[ response ].history = history
				
				# Recursively train this child node
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, fmt_str, verbose )