package com.myasuka.pca_detector

import java.io.{FileWriter, BufferedWriter}

import breeze.linalg.{DenseVector => BDV}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, Path, FileSystem}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import breeze.stats._


import scala.collection.mutable
import scala.util.Random

object PCASpark {
  def main(args: Array[String]) {
    if (args.length < 3){
      println("usage: PCASpark <rank> <input> <output>")
      System.exit(1)
    }
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val input = args(0)
    val output = args(1)
    val rank = args(2).toInt
    val movieIndexSet = new mutable.HashSet[Int]()
    val userIndexSet = new mutable.HashSet[Int]()
    val hdfsConf = new Configuration()
    val hdfs = FileSystem.get(hdfsConf)
    val inputStream = new FSDataInputStream(hdfs.open(new Path(input)))
    val inputData= scala.io.Source.fromInputStream(inputStream, "UTF-8").getLines().map{ line =>
      val info = line.split("::|\\s+")
      userIndexSet.+=(info(0).toInt)
      movieIndexSet.+=(info(1).toInt)
      (info(0).toInt, info(1).toInt, info(2).toDouble)
    }.toSeq
    println(s"inputData size: ${inputData.size}")
    val movieToIndex = movieIndexSet.toArray.sortWith(_ < _).zipWithIndex.toMap
    val userIndex = userIndexSet.toArray.sortWith(_ < _).zipWithIndex
    val userToIndex = userIndex.toMap
    val indexToUser = userIndex.map(t => (t._2, t._1)).toMap

    val distinctMovie = movieIndexSet.size
    println(s"distinctMovie: $distinctMovie")
    val rowLen = movieIndexSet.max
    println(s"rowLen: $rowLen")

    val distinctUser = userIndexSet.size
    println(s"distinctUser: $distinctUser")
    val userLen = userIndexSet.max
    println(s"userLen: $userLen")
    val dataRows = sc.parallelize(inputData, 32).groupBy(_._1).map{vec => {
      val (indices, values) = vec._2.map(e => (movieToIndex.get(e._2).get, e._3)).unzip
      val vector = BDV.zeros[Double](rowLen)
      val indArray = indices.toArray
      val valArray = values.toArray
      for ( i <- 0 until indices.size){
        vector(indArray(i)) = valArray(i)
      }
      val mea = mean(vector)
      //      SparseVector(distinctMovie, values).toDenseVector.toArray
      //      val std = stddev(SparseVector(distinctMovie, values).toDenseVector.toArray)
      val std = stddev(vector)
      if (std == 0){
        (userToIndex.get(vec._1).get, new SparseVector(distinctMovie, indices.toArray, values.toArray.map(t => 1.0)))
      }else
      //        (userToIndex.get(vec._1).get, new SparseVector(distinctMovie, indices.toArray, values.toArray.map(t => (t-mea)/std)))
        (userToIndex.get(vec._1).get, vector.map(t => (t - mea)/std))
    }}

    println(s"dataRows count: ${dataRows.count()}")
    //    val matrix = new RowMatrix(dataRows.map(_._2))
    //    println(s"matrix num rows: ${matrix.numRows()}")

    //    val svd = matrix.computeSVD(rank, true)
    //    val result = svd.U.rows.map( row => row.toArray).zip(dataRows.map(_._1)).map(line => {
    //      val data = line._1.zipWithIndex.filter(_._2 < pc).map(t => math.pow(t._1, 2)).sum
    //        (line._2, data)
    //      }).collect().sortWith(_._2 < _._2)

    //    val indexMatrix = new IndexedRowMatrix(dataRows.map(t => IndexedRow(t._1.toLong, t._2)))
    val indexMatrix = new IndexedRowMatrix(dataRows.map(t => IndexedRow(t._1.toLong, new DenseVector( t._2.asInstanceOf[BDV[Double]].data))))
    println(s"indexMatrix info, indexMatrix num rows: ${indexMatrix.numRows()},  indexMatrix num cols: ${indexMatrix.numCols()}")
    //    println(s"indexedMatrix num rows: ${indexMatrix.numRows()}, indexedMatrix num cols: ${indexMatrix.numCols()}")
    val svd = indexMatrix.computeSVD(rank, true)
    println("vector: " + svd.s.toArray.mkString(","))
    val result = svd.U.rows.map{row => {
      var sum = 0.0
      for (i <- 0 until rank){
        sum += math.pow(row.vector(i), 2)
      }
      (row.index, sum)
    }}.collect().sortWith(_._2 < _._2)


    val writer = new BufferedWriter(new FileWriter(output))
    result.foreach(pair => writer.write(s"${indexToUser.get(pair._1.toInt).get},${pair._2}\n"))
    writer.flush()
    writer.close()
  }

}