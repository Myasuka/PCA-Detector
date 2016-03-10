package com.myasuka.pca_detector;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDCli;

import java.io.*;
import java.util.*;

/**
 * Implementation of shilling detection algorithm in collaborative filtering, based on Mahout.
 * 1. use `GenerateData` class on ml-10M data to insert spam user data, and upload to HDFS
 * 2. convert data in HDFS from original text format to sequence file format as next step's input
 * 3. use SVD method in mahout to get U and sigma matrix
 * 4. compute the distance of the contribution with the corresponding 'user' id and return the suspected users
 */
public class PCAMahout {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("USAGE: com.myasuka.pca_detector.PCAMahout <training data set> <output path> <suspected ratio>");
            System.exit(-1);
        }
        String inputPath = args[0];
        String outputPath = args[1];
        double suspectedRatio = Double.parseDouble(args[2]);
        String tempPath = outputPath + "/temp";
        String rank = "10";
        String overSampling = "10";
        Configuration conf = new Configuration();
        String infoMapPath = tempPath + "/maps";

        final String[] svdPath = new String[3];
        svdPath[0] = tempPath + "/seqMatrix";
        svdPath[1] = tempPath + "/svdOut";
        svdPath[2] = tempPath + "/tmp";


        HashMap<Integer, Integer> userIdCountMap = new HashMap<Integer, Integer>();
        HashMap<Integer, Pair> movieRateMap = new HashMap<Integer, Pair>();
        // pre step: transform user ID and movie ID
        System.out.println("pre step: transform user ID and movie ID");
        HashMap<Integer, Integer> rowIndex2userId = new HashMap<Integer, Integer>();
        int rateCounts = transformUserIdMovieId(conf, inputPath, userIdCountMap, movieRateMap, rowIndex2userId, infoMapPath);
        int userIdSize = userIdCountMap.size();
        int movieIdSize = movieRateMap.size();
        System.out.println("user Id set size: " + userIdSize);
        System.out.println("movie Id set size: " + movieIdSize);

        // step 1: generate sequence file for the use of SVD(PCA)
        System.out.println("step 1: generate sequence file for the use of SVD(PCA)");
        GenSeqMatrix genSeqMatrix = new GenSeqMatrix();
        genSeqMatrix.run(inputPath, tempPath, infoMapPath, movieIdSize);

        // step 2: execute SVD on the sequence file matrix, only generate U and sigma folder
        System.out.println("step 2: execute SVD(PCA)");
        System.out.println("the rank is " + rank + ", and the oversampling is " + overSampling);
        String[] svdArgs = new String[]{"--input", svdPath[0], "--output", svdPath[1], "--tempDir", svdPath[2],
                "--rank", rank, "--oversampling", overSampling, "--computeV", "false", "--reduceTasks", "8", "--powerIter", "1"};
        SSVDCli ssvdCli = new SSVDCli();
        ssvdCli.run(svdArgs);

        // step 3: compute distance and filter out spam users
        System.out.println("step 3: compute distance and filter out spam users");
        int averageRateNum = rateCounts / userIdSize;
        System.out.println("the average rate movies num: " + averageRateNum);
        int suspectedUserNum = (int) (suspectedRatio * userIdSize);
        System.out.println("suspected User Num: " + suspectedUserNum);
        int[] minDisUsers = computeDistance(conf, svdPath[1] + "/U", rank, rowIndex2userId, suspectedUserNum);
        System.out.println("minDis users size :" + minDisUsers.length);
        HashSet<Integer> spamUsers = new HashSet<Integer>();
        for (int d : minDisUsers) {
            spamUsers.add(d);
        }

        // evaluation step: test the precision of this algorithm
        int correct = 0;
        for (int spamID : spamUsers) {
            // when using GenerateData to generate the fake data, users with ID larger than 71567 are the spam users
            // if you are using ml-100K data, this threshold should be 943
            if (spamID > 71567) {
                correct++;
            }
        }
        System.out.println("spam users: " + correct + ", whole predict users: " + spamUsers.size()
                + ", precision: " + (double) correct / (double) spamUsers.size());
    }

    /**
     * Due to the actual row or column index in the matrix is not equal to the userId or movieId, we need to compress the whole matrix
     * to reduce the computation cost in the next steps
     *
     * @param conf
     * @param trainSetPath
     * @param userIdCountMap
     * @param movieRateMap
     * @param rowIndex2userId
     * @param mapPath
     * @throws IOException
     */
    private static int transformUserIdMovieId(Configuration conf, String trainSetPath,
                                              HashMap<Integer, Integer> userIdCountMap, HashMap<Integer, Pair> movieRateMap,
                                              HashMap<Integer, Integer> rowIndex2userId,
                                              String mapPath) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fs.open(new Path(trainSetPath))));
        String line;
        int rateCounts = 0;
        while ((line = bufferedReader.readLine()) != null) {
            String[] info = line.split("::|\\s+");
            int userId = Integer.parseInt(info[0]);
            int movieId = Integer.parseInt(info[1]);
            double rate = Double.parseDouble(info[2]);
            if (movieRateMap.containsKey(movieId)) {
                Pair pair = movieRateMap.get(movieId);
                double v = rate + (Double) pair.getFirst();
                int count = (Integer) pair.getSecond() + 1;
                Pair<Double, Integer> pairVal = new Pair<Double, Integer>(v, count);
                movieRateMap.put(movieId, pairVal);
            } else {
                Pair<Double, Integer> pairVal = new Pair<Double, Integer>(rate, 1);
                movieRateMap.put(movieId, pairVal);
            }
            if (userIdCountMap.containsKey(userId)) {
                int count = userIdCountMap.get(userId);
                count++;
                userIdCountMap.put(userId, count);
            } else userIdCountMap.put(userId, 1);
//            movieIdSet.add(movieId);
            rateCounts++;
        }
        bufferedReader.close();
        BufferedWriter user2RowIndexWriter = new BufferedWriter(
                new OutputStreamWriter(fs.create(new Path(mapPath + "/user2RowIndex"))));
        BufferedWriter movie2ColIndexWriter = new BufferedWriter(
                new OutputStreamWriter(fs.create(new Path(mapPath + "/movie2ColIndex"))));
        BufferedWriter movieAverageRateWriter = new BufferedWriter(
                new OutputStreamWriter(fs.create(new Path(mapPath + "/movieAvgRates"))));

        Iterator<Integer> userIdIterator = userIdCountMap.keySet().iterator();
        Iterator<Map.Entry<Integer, Pair>> movieIdIterator = movieRateMap.entrySet().iterator();

        int userIndex = 0;
        while (userIdIterator.hasNext()) {
            int userId = userIdIterator.next();
            user2RowIndexWriter.write(userId + "," + userIndex + "\n");
            rowIndex2userId.put(userIndex, userId);
            userIndex++;
        }

        int movieIndex = 0;
        while (movieIdIterator.hasNext()) {
            Map.Entry<Integer, Pair> entry = movieIdIterator.next();
            movie2ColIndexWriter.write(entry.getKey() + "," + movieIndex + "\n");
            double rate = (Double) entry.getValue().getFirst() / (Integer) entry.getValue().getSecond();
            movieAverageRateWriter.write(entry.getKey() + "," + rate + "\n");
            movieIndex++;
        }

        user2RowIndexWriter.flush();
        movie2ColIndexWriter.flush();
        movieAverageRateWriter.flush();
        user2RowIndexWriter.close();
        movie2ColIndexWriter.close();
        movieAverageRateWriter.close();
        fs.close();
        return rateCounts;
    }

    /**
     * compute the distance of the contribution with the corresponding 'user' id and return the num of suspected users
     *
     * @param conf
     * @param path
     * @param ranks
     * @param rowIndex2userId
     * @param suspectedUserNum
     * @throws IOException
     */
    private static int[] computeDistance(Configuration conf, String path, String ranks,
                                         HashMap<Integer, Integer> rowIndex2userId,
                                         int suspectedUserNum) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] statuses = fs.listStatus(new Path(path), hiddenFileFilter);
        HashMap<Integer, Double> userCoefficients = new HashMap<Integer, Double>();
        int count = 0;
        for (int i = 0; i < statuses.length; i++) {
            SequenceFile.Reader reader = new SequenceFile.Reader(conf,
                    SequenceFile.Reader.file(statuses[i].getPath()));
            IntWritable key = new IntWritable();
            VectorWritable value = new VectorWritable();
            while (reader.next(key, value)) {
                count++;
                DenseVector denseVector = (DenseVector) value.get();
                double sum = 0;
                for (int j = 0; j < Integer.parseInt(ranks); j++) {
                    sum += Math.pow(denseVector.get(j) * 100, 2);
                }
                userCoefficients.put(key.get(), sum);
            }
            reader.close();
        }
        ValueComparator vc = new ValueComparator(userCoefficients);
        TreeMap<Integer, Double> sortedMap = new TreeMap<Integer, Double>(vc);
        sortedMap.putAll(userCoefficients);
        System.out.println("user counts: " + count);
        System.out.println("user map size: " + userCoefficients.size());

        int[] suspectedUsers = new int[suspectedUserNum];
        Iterator<Map.Entry<Integer, Double>> iterator = sortedMap.entrySet().iterator();
        for (int i = 0; i < suspectedUserNum; i++) {
            suspectedUsers[i] = rowIndex2userId.get(iterator.next().getKey());
        }
        return suspectedUsers;
    }

    private static final PathFilter hiddenFileFilter = new PathFilter() {
        public boolean accept(Path p) {
            String name = p.getName();
            return !name.startsWith("_") && !name.startsWith(".");
        }
    };
}

class ValueComparator implements Comparator<Integer> {

    Map<Integer, Double> base;

    public ValueComparator(Map<Integer, Double> base) {
        this.base = base;
    }

    // Note: this comparator imposes orderings that are inconsistent with equals.
    public int compare(Integer a, Integer b) {
        if (base.get(a) >= base.get(b)) {
            return 1;
        } else {
            return -1;
        } // returning 0 would merge keys
    }
}
