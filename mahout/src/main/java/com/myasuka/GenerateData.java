package com.myasuka;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class GenerateData {
    // below is the status of ml-100k, please refer to http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
//    private static final long MOVIES = 1682;
//    private static final long USERS = 943;
    // below is the status of ml-10M, please refer to http://files.grouplens.org/datasets/movielens/ml-10m-README.html
    private static final long MOVIES = 10681;
    private static final long USERS = 71567;

    private static void spamUser(String[] args) throws IOException {
        // parse input args
        String inputPath = args[1];
        String outputPath = args[2];
        double attackPercentage = Double.parseDouble(args[3]);
        double filterPercentage = Double.parseDouble(args[4]);
        double randomPercentage = Double.parseDouble(args[5]);
        long randomSeed = System.currentTimeMillis();
        if (args.length > 6) {
            randomSeed = Long.parseLong(args[6]);
        }
        Random rnd = new Random(randomSeed);

        // calculate how many people are bad
        int badUserNum = (int) (attackPercentage * 0.01 * USERS);
        int filterMovieNum = (int) (filterPercentage * 0.01 * MOVIES);
        int randomAttackNum = (int) (randomPercentage * 0.01 * badUserNum);
        System.out.println("badUserNum: " + badUserNum);
        System.out.println("randomAttackNum: " + randomAttackNum);
        System.out.println("filterMovieNum: " + filterMovieNum);
        // statistics
        Map<Integer, Integer> movieToIndex = new HashMap<Integer, Integer>();
        Map<Integer, Integer> indexToMovie = new HashMap<Integer, Integer>();
        DescriptiveStatistics[] averageStats = new DescriptiveStatistics[(int) MOVIES];
        for (int i = 0; i < MOVIES; i++) {
            averageStats[i] = new DescriptiveStatistics();
        }
        DescriptiveStatistics totalStats = new DescriptiveStatistics();

        // traverse the data
        BufferedReader in = new BufferedReader(new FileReader(inputPath));
        BufferedWriter out = new BufferedWriter(new FileWriter(outputPath));
        String line = "";
        int lineNum = 0;
        int movieIndex = 0;
        line = in.readLine();
        while (line != null) {
            // split by "\t" is the ml-100k file
            String[] entry = line.split(":: || \\t");
            int movieId = Integer.parseInt(entry[1]);
            double rating = Double.parseDouble(entry[2]);
            if (!movieToIndex.containsKey(movieId)) {
                movieToIndex.put(movieId, movieIndex);
                indexToMovie.put(movieIndex, movieId);
                movieIndex++;
            }
            averageStats[movieToIndex.get(movieId)].addValue(rating);
            totalStats.addValue(rating);

            lineNum++;
            out.write(entry[0] + "::" + entry[1] + "::" + entry[2] + "::" + entry[3] + "\n");
            line = in.readLine();
        }
        in.close();

        int targetMovieIndex = rnd.nextInt((int) MOVIES);
        System.out.println("target movie: " + indexToMovie.get(targetMovieIndex));
        // random attack
        double totalAverageRate = totalStats.getMean();
        double totalStd = totalStats.getStandardDeviation();
        for (int i = 0; i < randomAttackNum; i++) {
            int userId = (int) (USERS + i + 1);
            int movies = 1;
            out.write(userId + "::" + indexToMovie.get(targetMovieIndex) + "::5::000000000\n");
            Set<Integer> movieIndexes = new HashSet<Integer>();
            while (movies < filterMovieNum) {
                movieIndex = rnd.nextInt((int) MOVIES);
                if (!movieIndexes.contains(movieIndex) && indexToMovie.containsKey(movieIndex) && movieIndex != targetMovieIndex) {
                    movieIndexes.add(movieIndex);
                    String rating = doubleToRating10M(totalStd * rnd.nextGaussian() + totalAverageRate);
                    // when coming across ml-100k data, use below code
                    //String rating = Integer.toString((int) Math.round(totalStd * rnd.nextGaussian() + totalAverageRate));
                    out.write(userId + "::" + indexToMovie.get(movieIndex) + "::" + rating + "::868245920\n");
                    movies++;
                }
            }
        }

        // average attack
        for (int i = randomAttackNum; i < badUserNum; i++) {
            int userId = (int) (USERS + i + 1);
            int movies = 1;
            out.write(userId + "::" + indexToMovie.get(targetMovieIndex) + "::5::000000000\n");
            Set<Integer> movieIndexes = new HashSet<Integer>();
            while (movies < filterMovieNum) {
                movieIndex = rnd.nextInt((int) MOVIES);
                if (!movieIndexes.contains(movieIndex) && indexToMovie.containsKey(movieIndex) && movieIndex != targetMovieIndex) {
                    double averageRate = averageStats[movieIndex].getMean();
                    double stdRate = averageStats[movieIndex].getStandardDeviation();
                    movieIndexes.add(movieIndex);
                    String rating = doubleToRating10M(stdRate * rnd.nextGaussian() + averageRate);
                    //String rating = Integer.toString((int)Math.round(stdRate * rnd.nextGaussian() + averageRate));
                    out.write(userId + "::" + indexToMovie.get(movieIndex) + "::" + rating + "::868245920\n");
                    movies++;
                }
            }
        }

        out.close();
    }

    private static String doubleToRating10M(double d) {
        if (d >= 0 && d < 0.75) {
            return "0.5";
        } else if (d >= 0.75 && d < 1.25) {
            return "1";
        } else if (d >= 1.25 && d < 1.75) {
            return "1.5";
        } else if (d >= 1.75 && d < 2.25) {
            return "2";
        } else if (d >= 2.25 && d < 2.75) {
            return "2.5";
        } else if (d >= 2.75 && d < 3.25) {
            return "3";
        } else if (d >= 3.25 && d < 3.75) {
            return "3.5";
        } else if (d >= 3.75 && d < 4.25) {
            return "4";
        } else if (d >= 4.25 && d < 4.75) {
            return "4.5";
        } else {
            return "5";
        }
    }


    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            printUsage();
            System.exit(-1);
        }
        if ("sparse".equals(args[0])) {
            sparse(args);
        } else if ("spamUser".equals(args[0])) {
            spamUser(args);
        } else {
            printUsage();
            System.exit(-1);
        }
    }

    private static void printUsage() {
        System.out.println("Usage: java GenerateData");
        System.out.println("       sparse <inputPath> <outputPath> <sparsePercentage> [randomSeed]");
        System.out.println("       spamUser <inputPath> <outputPath> <attackPercentage> " +
                "<filterPercentage> <randomPercentage> [randomSeed]");
    }

    private static void sparse(String[] args) throws IOException {
        // parse input args
        String inputPath = args[1];
        String outputPath = args[2];
        double sparsePercentage = Double.parseDouble(args[3]);
        long randomSeed = System.currentTimeMillis();
        if (args.length > 4) {
            randomSeed = Long.parseLong(args[4]);
        }

        BufferedReader in = new BufferedReader(new FileReader(inputPath));
        int lineNum = 0;
        while (in.readLine() != null) {
            lineNum++;
        }
        in.close();

        // calculate how many records need to be deleted
        int needDeleted = (int) (lineNum - sparsePercentage * 0.01 * USERS * MOVIES);
        if (needDeleted < 0) {
            needDeleted = 0;
        }

        // set deleted flags
        Random rnd = new Random(randomSeed);
        boolean[] delete = new boolean[lineNum];
        while (needDeleted > 0) {
            int index = rnd.nextInt(lineNum);
            if (!delete[index]) {
                delete[index] = true;
                needDeleted--;
            }
        }

        // traverse the data
        in = new BufferedReader(new FileReader(inputPath));
        BufferedWriter out = new BufferedWriter(new FileWriter(outputPath));
        String line = in.readLine();
        lineNum = 0;
        while (line != null) {
            if (!delete[lineNum]) {
                out.write(line + "\n");
            }
            line = in.readLine();
            lineNum++;
        }
        in.close();
        out.close();
    }
}
