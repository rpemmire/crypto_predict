import lzma
import pandas as pd
import numpy as np

'''
This script takes raw bitcoin blockchain data from multiple files, and out of that data, creates edge data for the first
100k sender receiver pairs in our data.
'''

def process_transaction(transactionid, inputs, outputs):
	#This method turns a list of m inputs and n outputs for a given transaction to weighted edge data

	#for each input, calculate their proportion of contribution to the given transaction
	inputs = pd.DataFrame(inputs).astype(float)
	inputs[5] = inputs[5]/inputs[5].sum()

	#draw an edge from an input to each output, with weight as the sum received by that output times proportion of contribution
	outputs = pd.DataFrame(outputs).astype(float)


	#inputs : txID, input_seq, prev_txID, prev_output_seq, addrID, sum
	#transaction outputs; format: txID, output_seq, addrID, sum
	#format to txID, in_addr, out_addr, weight

	edges = []
	for index, input_row in inputs.iterrows():
		for index, output_row in outputs.iterrows():
			weight = output_row[3]*input_row[5]
			in_addr = input_row[4]
			out_addr = output_row[2]
			if int(in_addr) == -1 or int(out_addr) == -1:
				#error handling for addresses, in first 100k, never got an error
				print("address -1")
			edges.append([int(transactionid), int(in_addr), int(out_addr), int(weight)])

	return edges


def check_numPairs(edgelist, goal_num = 100824):
	#this method checks the number of sender receiver pairs in the data to see
	#whether the program should still read in more data. We also print out number
	#of unique transactions and number of unique accounts. These numbers were slightly off
	#from the paper

	edgelist = np.concatenate(np.array(edgelist), 0)

	x = edgelist[:, 1]
	y = edgelist[:, 2]

	tuples = list(zip(x,y))
	numpairs = len(set(tuples))

	numtransactions = len(list(set(edgelist[:, 0])))
	numaccounts = len(set(list(x) + list(y)))

	if numpairs >= goal_num:
		#this stops the program
		return True

	print(numpairs, numtransactions, numaccounts)

	return False



#get all transactions and map each id with more than 0 inputs (non-mining) to num_inputs and num_outputs
trans = lzma.open('tx.dat.xz', mode='rt')

inputs = lzma.open('txin.dat.xz', mode='rt')
input_txnum = 0
myinput = None

outputs = lzma.open('txout.dat.xz', mode='rt')
output_txnum = 0
myoutput = None

alledges = []

#This while loop iterates through all three files to create edge data for specific transactions.
while True:

	#get a specific transaction
	transaction = trans.readline().strip().split("\t")
	transactionid, blockid, num_inputs, num_outputs = transaction[0], transaction[1], transaction[2], transaction[3]
	myinputs = []
	myoutputs = []

	#read into txin and txout to match transaction with inputs, outputs, and weights
	if int(num_inputs) >0:

		while True:
			#if input transaction id is = to our current, add it, read another line
			if int(input_txnum) == int(transactionid):
				myinputs.append(myinput)
				myinput = inputs.readline().strip().split("\t")
				input_txnum = myinput[0]

			#elif it is greater, break
			elif int(input_txnum) > int(transactionid):
				break

			#elif it is less or none, read another line
			elif int(input_txnum) < int(transactionid) or input_txnum == None:
				myinput = inputs.readline().strip().split("\t")
				input_txnum = myinput[0]


		while True:

			#if output transaction id is = to our current, add it
			if int(output_txnum) == int(transactionid):
				myoutputs.append(myoutput)
				myoutput = outputs.readline().strip().split("\t")
				output_txnum = myoutput[0]

			#elif it is greater, break
			elif int(output_txnum) > int(transactionid):
				break

			#elif it is less or none, read another line
			elif int(output_txnum) < int(transactionid) or int(output_txnum) == None:
				myoutput = outputs.readline().strip().split("\t")
				output_txnum = myoutput[0]

		#This checks if we have the same amount of inputs and outputs as specified by the transaction
		if len(myinputs) != int(num_inputs) or len(myoutputs) != int(num_outputs):
			print('uh oh')

		transactionedges = process_transaction(transactionid, myinputs, myoutputs)
		alledges.append(transactionedges)

		if check_numPairs(alledges) == True:
			break

#save file
edgelist = np.concatenate(alledges, 0)
edgelist = edgelist.astype(int)
np.savetxt('first100k.dat.xz', edgelist, fmt='%d')
