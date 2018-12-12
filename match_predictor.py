# -*- coding: utf-8 -*-

import math
import random

from sklearn.model_selection import StratifiedKFold

import sqlite3
import csv
from datetime import datetime

fieldnames = ['home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']

def split_lines(input, seed, output1, output2):
  """Distributes lines of 'input' to 'output1' and 'output2' pseudo-randomly."""
  random.seed(seed)
  out1 = open(output1, 'w')
  out2 = open(output2, 'w')
  for line in open(input, 'r').readlines():
    if random.randint(0, 1):
      out1.write(line)
    else:
      out2.write(line)

def getTeamName(co, team_api_id):
  print("load sqlite...")
  
  cursor = co.cursor()

  rows = cursor.execute("SELECT * FROM Team WHERE team_api_id = " + str(team_api_id)).fetchall()

  return rows[0][3]

def getTeamID(co, team_short_name):
  print("load sqlite...")
  
  cursor = co.cursor()

  rows = cursor.execute("SELECT * FROM Team WHERE team_short_name = '" + team_short_name.upper() + "'").fetchall()

  return rows[0][1]

def displayAllTeam(co):
  cursor = co.cursor()
  rows = cursor.execute("SELECT * FROM Team").fetchall()
  for x in rows:
    print(x[4] + ': ' + x[3])

# recuperer tous les matchs
# filtrer les informations
# ecrire en csv
def sqliteToFilteredCSV(co, output, limit):
  print("load sqlite...")
  
  cursor = co.cursor()

  rows = cursor.execute("SELECT * FROM Match LIMIT " + str(limit)).fetchall()

  columns = cursor.execute("PRAGMA table_info(Match)").fetchall()
  
  #cleaning
  fields = [[c[1], True] for c in columns]
  l=0
  for e in rows:
    removed = False
    for i, c in enumerate(e):
      if c == None:
        fields[i][1] = False
        removed =True
    if removed:
      l = l + 1

  print([f[0] for f in fields if f[1]])
  print("nb data: "+ str(len(rows)) + " - good data: " +str(l))

  with open(output, "w") as csvfile:
    wb = csv.DictWriter(csvfile, fieldnames)

    for x in rows:
      ll = {}
      for i, y in enumerate(x):
        if columns[i][1] in fieldnames:
          ll[columns[i][1]] = y
      wb.writerow(ll)

# CopiÃ© depuis td7.py
def read_data(filename):
  """Reads a breast-cancer-diagnostic dataset, like wdbc.data.

  Args:
    filename: a string, the name of the input file.
  Returns:
    A pair (X, Y) of lists:
    - X contient la liste des informations d'un matchs (equipes, dates...)
    - Y contient une liste des resultats d'un match 
      [(score domicile, score exterieur), ...] 
  """
  X = []
  Y = []
  for line in open(filename, 'r').readlines():
    fields = line.split(',')
    X.append([int(x) for x in fields[0:2]])
    if fields[2] > fields[3]:
      Y.append(1)
    if fields[2] < fields[3]:
      Y.append(-1)
    else:
      Y.append(0)
    # Y.append([int(x) for x in fields[2:4]])
  return X, Y


def simple_distance(data1, data2):
  return math.sqrt(math.pow(data1[1] - data2[1], 2) + math.pow(data1[0] - data2[0], 2))

def k_nearest_neighbors(x, match, dist_function, k):
  """Retourne la liste des matches les plus proches
  """ 
  distance_and_indices = [(dist_function(match[i], x), i) for i in range(len(match))]
  distance_and_indices.sort()
  return [di[1] for di in distance_and_indices[:k]]

def match_result_knn(x, train_x, train_y, dist_function, k):
  """predit le resultat d'un match
  """
  neigh = k_nearest_neighbors(x, train_x, dist_function, k)
  return sum([train_y[i] for i in neigh]) / len(neigh)

def eval_match_classifier(train_x, train_y, test_x, test_y, classifier, dist_function, k):
  """Evaluates a cancer KNN classifier.
  """
  num_trials = len(test_x)
  num_errors = 0.0
  for i in range(num_trials):
    if classifier(test_x[i], train_x, train_y, dist_function, k) != test_y[i]:
      num_errors += 1
  return num_errors / num_trials

def find_best_k(train_x, train_y, dist_function):
  """Uses cross-validation (10 folds) to find the best K for is_cancerous_knn().
  """
  def sampled_range(mini, maxi, num):
    if not num:
      return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out
  num_splits = 10
  best_k = None
  lowest_error = float('inf')
  for k in sampled_range(1, len(train_x) // 2, 30):
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=0)
    error = 0
    for train_indices, test_indices in skf.split(train_x, train_y):
      subtrain_x = [train_x[i] for i in train_indices]
      subtrain_y = [train_y[i] for i in train_indices]
      subtest_x = [train_x[i] for i in test_indices]
      subtest_y = [train_y[i] for i in test_indices]
      error += eval_match_classifier(subtrain_x, subtrain_y, subtest_x, subtest_y,
                                      match_result_knn, dist_function, k)
    # end for
    error /= num_splits
    if error < lowest_error:
      best_k = k
      lowest_error = error
  print('Obtained error rate %f with K = %d' % (lowest_error, best_k))
  return best_k

if __name__ == '__main__':
  def predictMatch(co, train_file, home, away):
    train = read_data(train_file)

    k = 5 #find_best_k(train[0], train[1], simple_distance)
    print(k)
    ma = [home, away]

    print("predication pour match: " + getTeamName(co, ma[0]) + " -" +getTeamName(co, ma[1]))
    
    rest = match_result_knn(ma, train[0], train[1], simple_distance, k)
    
    print("resultat du match: " + str(rest))
  

  with sqlite3.connect("database.sqlite") as co:
    csvfilenames = "data.csv"
    
    sqliteToFilteredCSV(co, csvfilenames, -1)
    
    # displayAllTeam(co)
    '''
    print('entrer abreviation des equipes')
    print('home>', end='')
    home = getTeamID(co, "psg") # input())
    print('away>', end='')
    away = getTeamID(co, "yb") # input())

    predictMatch(co, csvfilenames, home, away)
    '''