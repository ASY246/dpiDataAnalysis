/**
  * Created by ASY on 2016/12/26.
  * 用户搜索，及浏览数据分析
  */
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import java.util.regex.Matcher
import java.util.regex.Pattern


object DPISearchData_4G{

  def getDPIFromHDFS = (sc:SparkContext) => {
    /***
      * 1.默认读入分区较多，分成200个分区无shuffle，同时num-executors*executor-cores = 100*2, 并行度较高
      * 2.选取有效字段：宽带账号，App名称，网站名称，搜索关键字，终端类型，url，UA
      * 3.江苏一小时上网数据 0.001sample
      */
    sc.hadoopConfiguration.setBoolean("mapred.output.compress", false)
    val DPI_raw = sc.textFile("hdfs://ns/data/hjpt/edm/ilpf/comm_new/itf_4gdpi_mbl/jiangsu/20170326/*").coalesce(200).map(line => line.split('|'))
    //    val DPI_raw = sc.textFile("")
    val account_url = DPI_raw.filter(line => line.size>54).map(line => (line(0))).distinct.count()

    val account_url_output = account_url.map(line => (line._1 + "\t" + line._2 + '\t' + line._3 + '\t' + line._4 + '\t' + line._5 + '\t' + line._6 + '\t' + line._7) )

    account_url_output.sample(false, 0.001).repartition(1).saveAsTextFile("hdfs:///user/dm/asy/jiangsu_dpi_2017020820")

  }

  def getSearchData = (sc:SparkContext) => {
    /***
      * 统计用户搜索数据,共有91381665大概一亿条DPI数据,其中有搜索记录的大概有9560078,有效的中文搜索记录大概占其中的一半
      *
      */
    sc.hadoopConfiguration.setBoolean("mapred.output.compress", false)

    val DPI_raw = sc.textFile("/data/hjpt/edm/fix/itf_fix_dpi/jiangsu/20170208/20").coalesce(200).map(line => line.split('|'))

    val account_url = DPI_raw.filter(line => line.size>27).map(line => (line(0),line(5),line(9),line(11),line(12),line(24),line(26)))

    val account_url_output = account_url.filter(line => line._4.length() > 0).map(line => line._4)

    account_url_output.count()

  }

  /*
    def extractValue = (url:String) => {
      val urlObject = new UrlFramework(url)
      val Ch_res = urlObject.resValue
      Ch_res
    }
    def isContainChinese = (urlParse: String) => {
      val p = Pattern.compile("[\u4e00-\u9fa5]")
      val m = p.matcher(urlParse)
      var res = false
      if (m.find()) res = true
      res
    }
  */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("DPISearchData").setMaster("local[*]")
    val sc = new SparkContext(conf)
    //取江苏一天的固网DPI数据，后续还要加上移动DPI的数据
    val account_url = sc.textFile("D:\\DPIDataAnalysis\\data\\")

    //    val urlParse = account_url.map(line => (line._1,(line._2))).map(line => (line._1,extractValue(line._2))).filter(line => line._2.length >= 1)//返回宽带账号和url解析结果
    //    //加入对是否中文的判断

    //    val ChineseSearch = urlParse.filter{case(url, urlParse)=> isContainChinese(urlParse)}
    //      .map(line => (line._1,List(line._2)))
    //
    //      .reduceByKey(_:::_)

    print(account_url.count())
  }

}
