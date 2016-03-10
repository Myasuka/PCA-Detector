package com.myasuka.pca_detector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.log4j.Logger;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

public class GenSeqMatrix{


    public static class GenSeqMapper extends Mapper<LongWritable, Text, Text, Text> {
        //        private MultipleOutputs<Text, Text> mo;
        Text outKey = new Text("");
        String tempPath;
        String mapPath;
        HashMap<String, String> userId2RowIndexMap;
        HashMap<String, String> movieId2ColIndexMap;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            userId2RowIndexMap = new HashMap<String, String>();
            movieId2ColIndexMap = new HashMap<String, String>();
//            mo = new MultipleOutputs<Text, Text>(context);
            tempPath = context.getConfiguration().get("tempPath");
            mapPath = context.getConfiguration().get("mapPath");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fs.open(new Path(mapPath + "/user2RowIndex"))));
            System.out.println("reading path :" + mapPath + "/user2RowIndex");
            String line;
            while ((line = bufferedReader.readLine())!=null) {
                String[] info = line.split(",");
                userId2RowIndexMap.put(info[0], info[1]);
            }
            bufferedReader = new BufferedReader(new InputStreamReader(fs.open(new Path(mapPath + "/movie2ColIndex"))));
            System.out.println("reading path :" + mapPath + "/movie2ColIndex");
            while ((line = bufferedReader.readLine())!=null) {
                String[] info = line.split(",");
                movieId2ColIndexMap.put(info[0], info[1]);
            }
            bufferedReader.close();
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] info = value.toString().split("::");
            String content = info[0] + "," + info[1] + "," + info[2];
            outKey.set(content);
//            mo.write("output", outKey, NullWritable.get(), tempPath + "/trainSet/part");
//            context.write(new Text(info[0]), new Pair<Integer, Double>(Integer.parseInt(info[1]), Double.parseDouble(info[2])));
            context.write(new Text(userId2RowIndexMap.get(info[0])), new Text(movieId2ColIndexMap.get(info[1]) + "," + info[2]));
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
//            mo.close();
            super.cleanup(context);
        }
    }


    public static class GenSeqReducer extends Reducer<Text, Text, IntWritable, VectorWritable> {

        int movieLen;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            movieLen = context.getConfiguration().getInt("movies", 10681);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
//            Logger logger = Logger.getLogger(GenSeqReducer.class);
            ArrayList<Integer> indices = new ArrayList<Integer>();
            ArrayList<Float> elements = new ArrayList<Float>();
            Iterator<Text> iterator = values.iterator();
            while (iterator.hasNext()) {
                String[] info = iterator.next().toString().split(",");
                indices.add(Integer.parseInt(info[0]));
                elements.add(Float.parseFloat(info[1]));
            }
            double sum = 0;
            for (Float ele: elements){
                sum += ele;
            }
            double mean = sum / movieLen;
            double std = 0;
            for (Float ele: elements){
                std += Math.pow(ele - mean, 2);
            }
            std = (std + Math.pow(mean, 2)*(movieLen - elements.size()) )/ (movieLen - 1);
//            DescriptiveStatistics stat = new DescriptiveStatistics();
//            int len = indices.size();
//            for (int i = 0; i < len; i++) {
//                stat.addValue(elements.get(i));
//            }
            Vector vector = new DenseVector(movieLen);
            //below is z-score
            for (int i = 0; i < movieLen; i++) {
                vector.set(i, -mean / std);
            }
            for(int i=0;i< elements.size();i++){
                double val = (elements.get(i) - mean) / std;
                vector.set(indices.get(i), val);
//              vector.set(indices.get(i), elements.get(i)) ;
//              System.out.println("index: " + indices.get(i) + "  element: " + elements.get(i));

            }
            // maybe use DenseVector ?
            context.write(new IntWritable(Integer.parseInt(key.toString())), new VectorWritable(vector));
        }
    }

    public void run(String trainSet, String output, String mapPath, int movieIdSize) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        conf.set("tempPath", output);
        conf.set("mapPath", mapPath);
        conf.setInt("movies", movieIdSize);

        Job job = Job.getInstance(conf, "generate matrix");

//        job.addCacheFile(new Path(mapPath).toUri());
        System.out.println("the mapPath is " + mapPath);
        job.setMapperClass(GenSeqMapper.class);
        job.setReducerClass(GenSeqReducer.class);

        job.setJarByClass(GenSeqMatrix.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(VectorWritable.class);

        MultipleOutputs.addNamedOutput(job, "output", TextOutputFormat.class, Text.class, Text.class);
        FileInputFormat.addInputPath(job, new Path(trainSet));
        FileOutputFormat.setOutputPath(job, new Path(output + "/seqMatrix"));
        job.waitForCompletion(true);

    }
}
