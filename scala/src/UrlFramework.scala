import java.net.{URLEncoder,URLDecoder}
/**
  * Created by ASY on 2016/6/25.
  * 输入：url字符串
  * 根据规则解析出搜索引擎对应的url，并进行解码
  */
class UrlFramework(var url: String) {
  private val keys="wd=,q=,word=,query=,w=,keyword=,key=,k=,kw=,qw=,search/,search="
  //url = "http://m.baidu.com/su?from=1001703a&wd=%E4%B8%AD%E5%9B%BD%E9%82%AE%E9%80%9F%E8%BF%8E"
/*
  private val hostArray = {
    var cateGory = ""
    var domain = ""
    if(url.split("//").size == 2){
      val hostArrayWithoutPro = url.split("//")(1)
      val address = hostArrayWithoutPro.split("/")(0)
      if(address.split("\\.").size > 1){
        cateGory = address.split("\\.")(0)
        domain = address.split("\\.")(1)
      }
    }
    List(cateGory,domain)
  }
*/
  private var firstDecValue = ""

  def resValue = {
    var res = ""
    for(key <- keys.split(",") if firstDecValue.equals("")){
      val regularKey = "[^\\w\\s]+" + key
      //val regularKey = "[^\\w\\s]+wd="
      val originValue = getOriginValue(regularKey)
//      val decodeValue = getDecodeValue(originValue)
//      if(!decodeValue.equals("")){
//        firstDecValue = decodeValue
//      }
//      res = firstDecValue
      res = originValue
    }
    res
  }
  //def getCate_Domain = this.hostArray
  def getOriginValue(regularkey:String): String = {
    val keySplitArray = url.toLowerCase().split(regularkey)
    //val originValue = keySplitArray(1).split("&")(0).split("/")(0)
    val originValue = if(keySplitArray.size > 1) keySplitArray(1).split("&")(0).split("/")(0)  else ""
    originValue
  }
  def getDecodeValue(originValue:String, enc: String = "UTF-8"): String = {
    if(!originValue.equals("")) {
      val originValueRep = originValue.replaceAll("%(?![0-9a-fA-F]{2})", "%25").replaceAll("\\+", "%2B")
      val systemCodeConfiguration = System.getProperty("file.encoding")
      val decodeValue = URLDecoder.decode(originValueRep, systemCodeConfiguration) //此处需要声明originValueRep编码规则吗
      decodeValue
    }
    else ""
  }

}
