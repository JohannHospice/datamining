# -*- coding: utf-8 -*-

import math
import random
from sklearn.model_selection import StratifiedKFold
import sqlite3
from datetime import datetime
import sys

def getTeam(co, team_short_name):
  cursor = co.cursor()
  rows = cursor.execute("SELECT t.team_api_id, date, buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth FROM Team t, Team_Attributes ta WHERE t.team_short_name = '" + team_short_name.upper() + "' AND t.team_api_id = ta.team_api_id").fetchall()
  return rows

def displayAllTeam(co):
  cursor = co.cursor()
  rows = cursor.execute("SELECT * FROM Team").fetchall()
  for x in rows:
    print(x[4] + ': ' + x[3])

# recuperer tous les matchs
# filtrer les informations
# ecrire en csv
def sqliteToFilteredCSV(co, output, limit):
  def strToDate(x):
    return datetime.strptime(x.split(' ')[0], '%Y-%m-%d')
  def nearest(items, pivot):
    return min(items, key=lambda x: abs(strToDate(x[1]) - pivot))

  print("load sqlite...")

  cursor = co.cursor()
  MR = cursor.execute("SELECT home_team_goal, away_team_goal, home_team_api_id, away_team_api_id, date FROM Match LIMIT " + str(limit)).fetchall()
  TR = cursor.execute("SELECT team_api_id, date, buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth FROM Team_Attributes").fetchall()
  
  cpt = 0
  with open(output, "w") as f:
    for matchRow in MR:
      score = [str(e) for e in matchRow[:2]]
      ht = [t for t in TR if t[0] == matchRow[2]]
      at = [t for t in TR if t[0] == matchRow[3]]

      # si equipe non trouvée
      if len(ht) <= 0 or len(at) <= 0:
        cpt += 1
        continue

      date = strToDate(matchRow[4])
      home = nearest(ht, date)
      away = nearest(at, date)
      match = score + [str(a) for a in home[1:] if str(a).isdigit()] + [str(a) for a in away[1:] if str(a).isdigit()]
      f.write(','.join(match) + '\n')
  print("# matchs ignorés: " + str(cpt))

def load(filename):
  X = []
  Y = []
  for line in open(filename, 'r').readlines():
    fields = line.split(',')
    X.append([int(x) for x in fields[2:]])
    Y.append(fields[0] > fields[1])
  print("# matchs chargés: " + str(len(X)))
  return X, Y


def distance(data1, data2):
  """ distance euclidienne entre 2 matchs
  """
  return math.sqrt(sum([math.pow(data1[i] - data2[i], 2) for i in range(16)]))

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
  num_win = sum([1 for i in neigh if train_y[i]])
  return num_win > k - num_win

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
  print('finding the best k...')
  num_splits = 10
  best_k = None
  lowest_error = float('inf')
  for k in sampled_range(1, len(train_x) // 2, 30):
    #print('.', end='')
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
  print()
  print('Obtained error rate %f with K = %d' % (lowest_error, best_k))
  return best_k


# MAIN


if __name__ == '__main__':
  def chooseTeam(co):
    print("abbreviation de l'équipe> ", end='')
    team_short_name = input()
    rows = getTeam(co, team_short_name)
    for i, t in enumerate(rows):
      print(str(i) + ": " + str(t))
    if len(rows) > 1:
      print("indice> ", end='')
      ti = rows[int(input())]
    else: 
      ti = rows[0]
    return list(map(int, ti[2:]))

  k = 10
  def createAndLoad(filename, limit):
    sqliteToFilteredCSV(co, filename, limit)
    return load(filename)

  with sqlite3.connect("database.sqlite") as co:
    csvfilename = "data.csv"
    import os.path
   
    if os.path.isfile(csvfilename):
      data = load(csvfilename)
    else:
      data = createAndLoad(csvfilename, 100)

    while True:
      print("0: Evaluer un match\t1: Assigner le K\t2: Afficher list des équipes\t3: Trouver le meilleur K\t4: Charger données")
      print("choix>", end='')
      i = int(input())
      if i == 0:
        print("equipe à domicile")
        t1 = chooseTeam(co)
        print("equipe exterrieur")
        t2 = chooseTeam(co)

        print("predication du match")
        rest = match_result_knn(t1 + t2, data[0], data[1], distance, k)
        if rest: 
          s = "l'équipe à domicile gagne"
        else:
          s = "l'équipe à domicile perd ou match null"

        print("resultat du match: " + s)
      elif i == 1:
        print("k>", end='')
        k = int(input())
      elif i == 2:
        print("liste des equipes")
        displayAllTeam(co)
      elif i == 3:
        k = find_best_k(data[0], data[1], distance)
      elif i == 4:
        print('limit>', end='')
        data = createAndLoad(csvfilename, int(input()))

      else: 
        m = list(map(int, getTeam(co, 'COR')[0][2:])) + list(map(int, getTeam(co, 'CEL')[0][2:]))
        rest = match_result_knn(m, data[0], data[1], distance, k)
        print(rest)
