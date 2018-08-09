import java.io.*;
import java.util.*;
import java.lang.*;
import java.security.*;

public class ABCSpeedup {
    private final int TIME_LIMIT = 10 * 60 * 1000;// 15 minutes
    private final double SCORE_MULTIPLIER = 1000000.0;

    public String checkData(String seed) {
        return "";
    }

    public String displayTestCase(String seed) {
        return "Seed: " + seed;
    }
	
    // X is the returned one, Y is the ground truth
	private double computeSimilarity(String[] X, String[] Y) {
		HashMap<String, ArrayList<String>> clusterX = new HashMap<String, ArrayList<String>>();
		HashMap<String, ArrayList<String>> clusterY = new HashMap<String, ArrayList<String>>();
		for (int i = 0; i < X.length; ++ i) {
			if (!clusterX.containsKey(X[i])) {
				clusterX.put(X[i], new ArrayList<String>());
			}
			clusterX.get(X[i]).add(Y[i]);
			
			if (!clusterY.containsKey(Y[i])) {
				clusterY.put(Y[i], new ArrayList<String>());
			}
			clusterY.get(Y[i]).add(X[i]);
		}
		
		long sameXsameY = 0, diffXsameY = 0, sameXdiffY = 0, diffXdiffY = 0;
		for (ArrayList<String> list : clusterX.values()) {
			HashMap<String, Integer> cnt = new HashMap<String, Integer>();
			for (int i = 0; i < list.size(); ++ i) {
				String other = list.get(i);
				if (!cnt.containsKey(other)) {
					cnt.put(other, 0);
				}
				cnt.put(other, cnt.get(other) + 1);
			}
			for (int c : cnt.values()) {
				sameXsameY += (long)c * (c - 1) / 2;
				sameXdiffY += (long)c * (list.size() - c);
			}
		}
		for (ArrayList<String> list : clusterY.values()) {
			HashMap<String, Integer> cnt = new HashMap<String, Integer>();
			for (int i = 0; i < list.size(); ++ i) {
				String other = list.get(i);
				if (!cnt.containsKey(other)) {
					cnt.put(other, 0);
				}
				cnt.put(other, cnt.get(other) + 1);
			}
			for (int c : cnt.values()) {
				diffXsameY += (long)c * (list.size() - c);
			}
		}
		long all = (long)X.length * (X.length - 1) / 2;
		diffXdiffY = all - sameXsameY - sameXdiffY - diffXsameY;
		
		long num = sameXsameY;
        
		long base = 0;
        for (ArrayList<String> list : clusterY.values()) {
            base += (long)list.size() * (Y.length - list.size());
        }
        base /= 2;
        
		double result = 0;
		if (all > 0 && num >= base) {
			result = (double)(num - base) / (all - base);
		}
		return result;
	}

    public double runTest(LongTest lt) {
		LongTest.WriterCache cache = lt.newCacheInstance();
        String seed = lt.getTest();
		String[] data = (String[]) cache.get(seed + ".json");
        String[] result = (String[]) cache.get(seed + ".out");
        
        cache.setMinimalVersion(1);

        //Run Solution
        long timeLeft = TIME_LIMIT;
        lt.setTimeLimit(TIME_LIMIT);
        lt.cluster(data);
        if (!lt.getStatus()) {
            lt.addFatalError("Error during the call to predictYield method.");
            return 0.0;
        }
        String[] rv = lt.getResult_cluster();
        if (!lt.getStatus()) {
        	lt.addFatalError("Error during the call to predictYield method.");
        	return 0.0;	
        }
        long timeSpent = lt.getTime();
        if (TIME_LIMIT - timeSpent < 10) {
			lt.addFatalError("Time limit exceeded!");
            return 0.0;
        }
        
		int max_cluster_id = 0;
        for (int i = 0; i < rv.length; ++i) {
            int cluster_id = Integer.parseInt(rv[i]);
			max_cluster_id = Math.max(max_cluster_id, cluster_id + 1);
        }
		if (max_cluster_id > data.length) {
			lt.addFatalError("Too many clusters! " + max_cluster_id + " cluster ids for only " + data.length + " data.");
            return 0.0;
		}
		
        int returned = rv.length, target = result.length;
		if (returned != target) {
			lt.addFatalError("Wrong size of returned results! " + target + " expected, but get " + returned + ".");
            return 0.0;
		}
		
		double similarity = computeSimilarity(rv, result);
        
        similarity = Math.max(similarity - 0.99, 0);
		if (timeSpent < 100) {
			timeSpent = 100;
		}
		return (similarity * similarity * similarity * similarity) / timeSpent;
    }

    public double[] score(double[][] scores) {
        int userCnt = scores.length;
        double[] res = new double[userCnt];
        if (userCnt == 0) return res;
		
        int caseCnt = scores[0].length;
        for (int caseNum = 0; caseNum < caseCnt; ++ caseNum) {
			double maximum = 0;
            for (int userNum = 0; userNum < userCnt; ++ userNum) {
				maximum = Math.max(maximum, scores[userNum][caseNum]);
            }
			for (int userNum = 0; userNum < userCnt; ++ userCnt) {
				double ratio = (scores[userNum][caseNum] / maximum);
				res[userNum] += ratio * SCORE_MULTIPLIER;
			}
        }

        for (int userNum=0; userNum < userCnt; userNum++) {
            res[userNum] /= (double) caseCnt;
        }

        return res;
    }
}
