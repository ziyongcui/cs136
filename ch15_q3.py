import pandas as pd
from math import sqrt
import numpy as np


##########################################################

# Python Program for Floyd Warshall Algorithm - based off of code from online
 
# Number of vertices in the graph
V = 25
 
# Define infinity as a large enough value. 
# Will be used for vertices not connected to each other
INF = 999999999
 
# Solves all pair shortest path
# via Floyd Warshall Algorithm
 
def floydWarshall(graph):
   
    """ dist[][] will be the output
       matrix that will finally
        have the shortest distances
        between every pair of vertices """
 
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
    return dist

# This code is contributed by Mythri J L

##########################################################

# load f1 results data
df = pd.read_csv('resultss.csv') 

# look at data from races with id 1-17 (all those from 2009 season)
test = pd.DataFrame()
for i in range(1,18):
    test = test.append(df[df['raceId'] == i])    
test = test.reset_index(drop=True)

# get all racer ids from 2009 season
racers = list(set(list(test['driverId'])))
num_racers = len(racers)

# racer dict to map racer id from original data to 0-24
racers_dict = {}
for c,r in enumerate(racers):
    racers_dict[r] = c

# create inverse of above dict
inverse_racers_dict = {v: k for k, v in racers_dict.items()}

# create driver id to name dict for printing out results at the end
df_drivers = pd.read_csv('drivers.csv')
driver_id_to_name = {}
for i in range(len(df_drivers)):
    driver_id_to_name[df_drivers['driverId'][i]] = df_drivers['driverRef'][i]

##########################################################

# PROBLEM 3(b)

# create defeat graph where graph[i][j] is the number of times racer j beat racer i that season (and INF otherwise)
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

# run and print out floyd warshall results to see if we satisfy the all pairs paths property
dist = floydWarshall(graph)
print('result from running floyd-warshall:')
for row in dist:
    print(row)

##########################################################

# PROBLEM 3(c)

# set up observations and not_last variables
# observations[i] is a list of drivers in order for race i
# not_last[i] is the number of times that i ranked above the last position in an observation
observations = {}
not_last = [0] * len(racers)
for i in range(1,18):
    data = df[df['raceId'] == i].reset_index(drop=True)
    drivers = list(data['driverId'])
    driver_ids = [racers_dict[d] for d in drivers]
    observations[i] = driver_ids
    for d in driver_ids[:-1]:
        not_last[d] += 1

# function that updates the scores
def update_scores(scores):
    new_scores = [0] * len(scores)
    
    # loop over each alternative
    for j in range(len(racers_dict)):

        # running sum for the denominator of our update step
        s = 0

        # double outer loop
        for i in range(1, len(observations)+1):
            for k in range(len(observations[i])-1):

                # effectively having delta=1 if alt j is ranked in pos k or lower for ith observation (and 0 o/w)
                if j in observations[i][k:]:
                    # sum up relevant old scores and update our outer running sum
                    inv_s = 0
                    for l in range(k, len(observations[i])):
                        inv_s += scores[observations[i][l]]
                    inv_s = 1 / inv_s
                    s += inv_s

        # set new score
        new_scores[j] = not_last[j] / s

    # normalize new scores
    normalizing_factor = sum(new_scores)
    for i in range(len(new_scores)):
        new_scores[i] /= normalizing_factor

    return new_scores
    
# function that runs the updated mm algorithm
def mm_alg():
    # initialize scores
    scores = [1/len(graph[0])] * len(graph[0])

    converged = False
    epsilon = .001

    while not converged:
        # get new scores
        new_scores = update_scores(scores)

        # get L2 diff between new and old scores
        diff = sum([abs(new_scores[i]-scores[i]) for i in range(len(scores))])
        diff = sqrt(diff)

        scores = new_scores

        # stop iterating if difference is below epsilon
        if diff < epsilon:
            converged = True
            
    return scores

# run the algorithm on the 2009 F1 data
final_scores = mm_alg()

# sort the scores to get the winning drivers and get their original ids
winning_drivers = np.argsort(final_scores)[::-1]
winning_drivers_original_ids = [inverse_racers_dict[d] for d in winning_drivers]

# print out names of drivers in final order
print([driver_id_to_name[w] for w in winning_drivers_original_ids])