from xo.game  import Game
from xo.board import Board
from xo       import ai
from xo       import arbiter
from xo.board import ncells
from xo.token import isempty, istoken, other_token
import xo

import numpy as np
import pickle
from pathlib import Path
import time

class Node():
	def __init__(self, board, player):
		assert isinstance(board, Board)
		self.player = player
		self.board = board
		self.info_set = str(board) + player
		self.num_actions = len(list(board.empty_spaces()))
		self.regretSum   = np.zeros(self.num_actions)
		self.strategy    = np.zeros(self.num_actions)
		self.strategySum = np.zeros(self.num_actions)

	def getStrategy(self, realizationWeight):
		assert isinstance(realizationWeight, float)

		normalizingSum = np.sum(np.maximum(self.regretSum, 0))
		if normalizingSum > 0:
			self.strategy = np.maximum(self.regretSum, 0) / normalizingSum
		else:
			self.strategy.fill(1.0 / self.num_actions)

		self.strategySum += realizationWeight * self.strategy
		return self.strategy

	def getAverageStrategy(self):
		normalizingSum = np.sum(self.strategySum)
		if normalizingSum > 0:
			return self.strategySum / normalizingSum
		else:
			return np.full(self.num_actions, 1.0 / self.num_actions)

	def __str__(self):
		return self.info_set + ": " + str(self.getAverageStrategy())

	def __repr__(self):
		return self.__str__()

	def __hash__(self):
		return hash(self.info_set)

node_map = {}

_maximum_depth = 9

def utility_score(outcome, depth):
	if outcome['reason'] == arbiter.REASON_WINNER or outcome['reason'] == arbiter.REASON_LOSER:
		return 2 * (_maximum_depth - depth) + _maximum_depth + 1
	elif outcome['reason'] == arbiter.REASON_SQUASHED:
		return depth
	else:
		raise ValueError('unexpected outcome: {}'.format(outcome))

PLAYER = {
	"x":  1,
	"o": -1
}


def cfr(board, player, t, p0, p1):
	outcome = arbiter.outcome(board, player)

	if ai._terminal(outcome):
		return PLAYER[player] * utility_score(outcome, t)

	info_set = str(board) + player

	if info_set not in node_map:
		node_map[info_set] = Node(board, player)
	
	node = node_map[info_set]

	strategy = node.getStrategy(p0 if player == "x" else p1)
	utility  = np.zeros(node.num_actions, dtype=np.float32)
	node_utility = 0


	for ((r,c), i) in zip(board.empty_spaces(), range(node.num_actions)):
		
		if player == "x":
			new_board = board.copy()
			new_board[r, c] = "x"
			utility[i] = cfr(new_board, "o", t + 1, p0 * strategy[i], p1)
		else:
			new_board = board.copy()
			new_board[r, c] = "o"
			utility[i] = cfr(new_board, "x", t + 1, p0, p1 * strategy[i])

		try:
			node_utility += strategy[i] * utility[i]
		except RuntimeWarning:
			print(node_utility)

	for i in range(node.num_actions):
		regret = utility[i] - node_utility
		node.regretSum[i] += regret * (p1 if player == "x" else p0)

	return node_utility

def train_cfr(board, iterations):
	util = 0.0
	for i in range(iterations):
		print("STARTING TRAINING {}".format(i))
		util += cfr(board, "x", 0, 1.0, 1.0)
	return util / iterations

def main():
	global node_map

	my_file = Path("./brain.pickle")
	if not my_file.is_file():
		b = Board.fromstring()
		ret = train_cfr(b, 5)
		with open('./brain.pickle', 'wb') as handle:
			pickle.dump(node_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print(ret)
	else:
		with open('./brain.pickle', 'rb') as handle:
			node_map = pickle.load(handle)


	g = Game()
	g.start('x')
	player = "x"
	while True:
		# Human Player Goes First
		if player == "x":
			x, y = input("Enter coordinates:").split(" ")
			outcome = g.moveto(int(x),int(y))
			if outcome["name"] == xo.game.EVENT_NAME_INVALID_MOVE:
				print("MOVE: {}, {}".format(x,y))
				print("INVALID MOVE")
				continue
			print(g.board.toascii())
			player = "o"
		else:
			time.sleep(.5)
			info_set = str(g.board) + "o"
			empty_spaces = np.array([[r,c] for (r,c) in g.board.empty_spaces()])

			node = node_map[info_set]
			xy = empty_spaces[np.random.choice(len(list(g.board.empty_spaces())), p=node.getAverageStrategy())]
			x, y = xy[0], xy[1]
			outcome = g.moveto(x, y)
			if outcome["name"] == xo.game.EVENT_NAME_INVALID_MOVE:
				print("MOVE: {}, {}".format(x, y))
				print("INVALID MOVE")
				continue
			print(g.board.toascii())
			player = "x"




	print("Average utility: {}".format(ret))
	print("Completed States: {}". format(X))


if __name__ == '__main__':
	main()
