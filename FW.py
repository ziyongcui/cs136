# Python Program for Floyd Warshall Algorithm
 
# Number of vertices in the graph
V = 25
 
# Define infinity as the large
# enough value. This value will be
# used for vertices not connected to each other
INF = 99999
 
# Solves all pair shortest path
# via Floyd Warshall Algorithm

import pandas as pd
 
def floydWarshall(graph):
   
    """ dist[][] will be the output
       matrix that will finally
        have the shortest distances
        between every pair of vertices """
    """ initializing the solution matrix
    same as input graph matrix
    OR we can say that the initial
    values of shortest distances
    are based on shortest paths considering no
    intermediate vertices """
 
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
 
    """ Add all vertices one by one
    to the set of intermediate
     vertices.
     ---> Before start of an iteration,
     we have shortest distances
     between all pairs of vertices
     such that the shortest
     distances consider only the
     vertices in the set
    {0, 1, 2, .. k-1} as intermediate vertices.
      ----> After the end of a
      iteration, vertex no. k is
     added to the set of intermediate
     vertices and the
    set becomes {0, 1, 2, .. k}
    """
    for k in range(V):
 
        # pick all vertices as source one by one
        for i in range(V):
 
            # Pick all vertices as destination for the
            # above picked source
            for j in range(V):
 
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j]
                                 )
    # printSolution(dist)
    return dist
 
 
# A utility function to print the solution
def printSolution(dist):
    print ("Following matrix shows the shortest distances\
 between every pair of vertices")
    for i in range(V):
        for j in range(V):
            if(dist[i][j] == INF):
                print ("%7s" % ("INF"),end=" ")
            else:
                print ("%7d\t" % (dist[i][j]),end=' ')
            if j == V-1:
                print ()
 
 
# Driver program to test the above program
# Let us create the following weighted graph
"""
            10
       (0)------->(3)
        |         /|\
      5 |          |
        |          | 1
       \|/         |
       (1)------->(2)
            3           """

import pandas as pd
df = pd.read_csv('resultss.csv') ## 
test = pd.DataFrame()
for i in range(1,18):
    test = test.append(df[df['raceId'] == i])
    
test = test.reset_index(drop=True)
racers = list(set(list(test['driverId'])))
num_racers = len(racers)
racers_dict = {}
for c,r in enumerate(racers):
    racers_dict[r] = c
graph = [[0 for col in range(num_racers)] for row in range(num_racers)]
for i in range(1,18):
    data = df[df['raceId'] == i].reset_index(drop=True)
    drivers = list(data['driverId'])
    for c,d in enumerate(drivers):
        for l in drivers[c+1:]:
            winner = racers_dict[d]
            loser = racers_dict[l]
            graph[loser][winner] += 1

for i in range(num_racers):
    for j in range(num_racers):
        if i != j and graph[i][j] == 0:
            graph[i][j] = INF


# graph = [[0, 5, INF, 10],
#          [INF, 0, 3, INF],
#          [INF, INF, 0,   1],
#          [INF, INF, INF, 0]
#          ]
# Print the solution
dist = floydWarshall(graph)
for row in dist:
    print(row)
# This code is contributed by Mythri J L


def update_scores(scores, n, wins):
    return


def mm_alg(graph):
    # n[i][j] contains the number of times that i and j have competed against eachother
    n = [[0]*len(graph[0]) for _ in range(len(graph[0]))]

    # wins[j] is the number of wins by j against other alternatives
    wins = [0] * len(graph[0])

    # setup n and wins
    for i in range(len(graph[0])):
        for j in range(len(graph[0])):
            n[i][j] = graph[i][j] + graph[j][i]
            wins[j] += graph[i][j]

    # initialize scores
    scores = [1/len(graph[0])] * len(graph[0])

    converged = False

    while not converged:
        new_scores = update_scores(scores, n, wins)
        
            
    return