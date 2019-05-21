#classes to handle building lists of edges

class Undirected:
	# The structure is: {source: [target, target, ...], ... }
	# instantiate: ddi_all = Undirected({})
	# get the edges: ddi_all.edges
	def __init__(self, edgeDict):
		self.edges = edgeDict

	def add(self, nodeA, nodeB):
		inA = False
		inB = False
		inAB = False
		inBA = False
		if nodeA in self.edges:
			inA = True
			if nodeB in self.edges[nodeA]:
				inAB = True
		if nodeB in self.edges:
			inB = True
			if nodeA in self.edges[nodeB]:
				inBA = True
		if not (inAB or inBA):
			if inA:
				self.edges[nodeA].add(nodeB)
			elif inB:
				self.edges[nodeB].add(nodeA)
			else:
				self.edges[nodeA] = set([nodeB])

	def count(self):
		c = 0
		for a in self.edges:
			c += len(self.edges[a])
		return c

	def list(self):
		result = []
		for a in self.edges:
			for b in self.edges[a]:
				result.append([a,b])
		return result

class Directed(object):
	"""docstring for Directed"""
	def __init__(self, edgeDict):
		self.edges = edgeDict

	def add(self, source, target):
		if source in self.edges:
			self.edges[source].add(target) #it's a set so the target will only be added if it's new
		else:
			self.edges[source] = set([target])

	def count(self):
		c = 0
		for a in self.edges:
			c += len(self.edges[a])
		return c
		
	def list(self):
		result = []
		for a in self.edges:
			for b in self.edges[a]:
				result.append([a,b])
		return result

#demo
# >>> from edges import Undirected
# >>> ed = Undirected({})
# >>> ed.add('a','b')
# >>> ed.add('b','c')
# >>> ed.add('a','m')
# >>> ed.count()
# 3
# >>> ed.list()
# [['a', 'b'], ['a', 'm'], ['b', 'c']]
# >>> 
