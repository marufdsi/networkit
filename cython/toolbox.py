"""
This module contains general purpose code.
"""


# batch processing


def batch(graphDir, match, format, function, outPath, header=None):
	"""
	Read graphs from a directory, apply a function and store result in CSV format.
	:param	graphDir	a directory containing graph files 
	:param	match		a pattern that must match the filename so the file is treated as a graph
	:param 	format		graph file format
	:param  function	any function from Graph to list/tuple of values
	:param	header		CSV file header
	"""
	with open(outPath, 'w') as outFile:
		writer = csv.writer(outFile, delimiter='\t')
		if header:
			writer.writerow(header)
		for root, _, filenames in os.walk(graphDir):
			for filename in filenames:
				if match in filename:
					print("processing {0}".format(filename))
					graphPath = os.path.join(root, filename)
					timer = stopwatch.Timer()
					G = graphio.readGraph(graphPath)
					timer.stop()
					row = function(G)
					row = [filename, timer.elapsed] + list(row)
					writer.writerow(row)
