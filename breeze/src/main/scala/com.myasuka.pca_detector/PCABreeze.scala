package com.myasuka.pca_detector

import scala.io.Source

import breeze.linalg.{DenseMatrix, SparseVector}
import breeze.linalg._
import breeze.stats._

object PCABreeze{
  def main(args: Array[String]) {
    if (args.length < 3){
      println("usage: PCABreeze <input> <output> <rank>")
      System.exit(1)
    }
    val input = args(0)
    val rank = args(2).toInt
    val lines = Source.fromFile(input).getLines()
    val movies = 1682
    val rows = lines.map{line =>
      val info = line.split("::")
      (info(0).toInt - 1, (info(1).toInt - 1, info(2).toDouble))
    }.toTraversable.groupBy(_._1).map( vecs => {
      val (indices, values) = vecs._2.map(t => (t._2._1, t._2._2)).unzip
      (vecs._1, (indices.toArray, values.toArray))
    })
    val users = rows.size
    println(s"users count: $users")
    val matrix = new DenseMatrix[Double](users, movies)
    for( i <- 0 until users){
      val vector = DenseVector.zeros[Double](movies)
      val sparseVector = rows.get(i).get
      for(j <- 0 until sparseVector._1.length){
        vector(sparseVector._1(j)) = sparseVector._2(j)
      }
      matrix(i,::) := vector.t
    }

    //use z-score
    for ( i <- 0 until users){
      val vec = matrix(i, ::).inner
      val mea = mean(vec)
      val std = stddev(vec)
      matrix(i, ::) := vec.map( t => (t - mea) / std).t
    }

    val u = svd(matrix).U
    println(s"u rows: ${u.rows}, u columns: ${u.cols}")
    val result = DenseVector.zeros[Double](users)
    for(i <- 0 until rank){
      result += u(::, i).map( t=> math.pow(t, 2))
    }
    println(s" sum of the result vector ${sum(result)} ")
    val r = 94
    val spam = result.toArray.zipWithIndex.
      sortWith(_._1 < _._1).take(r).filter(_._2 >= 943).size
    println(s"detect spam user $spam, whole predict user $r, the precision: ${spam/r.toDouble}")
  }

}
