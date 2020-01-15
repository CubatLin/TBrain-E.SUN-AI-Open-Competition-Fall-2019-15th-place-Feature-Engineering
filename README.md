玉山銀行人工智慧公開挑戰賽_2019秋季賽_真相只有一個_信用卡盜刷偵測-15th place(top1%) Feature Engineering
======================================
* **比賽網址：**[TBrain E.SUN AI Open Competition Fall 2019 ](https://tbrain.trendmicro.com.tw/Competitions/Details/10)

* **參賽成績：**
```
Team:菜雞互啄
Members:Alexi,Pat,Michael,Ming-Xiang,Ethan
F1 score prediction competition: 15th of 1366 teams
Creativity presentation competition: 2nd
```
* **參賽海報(閃電講2nd)：**[菜雞互啄_海報](https://github.com/CubatLin/TBrain-E.SUN-AI-Open-Competition-Fall-2019-15th-place-Feature-Engineering/blob/master/%E8%8F%9C%E9%9B%9E%E4%BA%92%E5%95%84_%E7%8E%89%E5%B1%B1%E9%8A%80%E8%A1%8C2019%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E7%A7%8B%E5%AD%A3%E5%85%AC%E9%96%8B%E6%8C%91%E6%88%B0%E8%B3%BD%E5%85%A8%E9%96%8B%E6%B5%B7%E5%A0%B1.pdf)
* **Feature Engineering：**[Python Code](https://github.com/CubatLin/TBrain-E.SUN-AI-Open-Competition-Fall-2019-15th-place-Feature-Engineering/blob/master/%E8%8F%9C%E9%9B%9E%E4%BA%92%E5%95%84_ModeCode.py)

Description 
===========
一卡在手，妙用無窮！ 在台灣，20歲以上持有信用卡人數超過六成。因信用卡具備高回饋、延遲付款以及付款便利等特性，使得信用卡成為人們支付時不可或缺的工具。不過隨著科技的日新月異，不肖分子也針對此支付模式衍生出新的犯罪手法，即「信用卡盜刷」。

面對盜刷，一般民眾除了可以透過經常對帳、防止卡片資訊外洩等方式來避免外，國內外銀行及發卡組織近年也開始運用機器學習演算法找出潛在的盜刷交易，並及早因應。然而，盜刷的樣態千百種，到底什麼才是足以判斷為盜刷的關鍵因子呢？

本次競賽提供去識別信用卡交易授權資料，希望大家集思廣益，一同「反盜刷」！不僅捍衛自己的資產，守護身邊親友的財富，更有機會獲得高額獎金！

本次競賽共包含兩場獨立賽事，分別為「線上對決–模型準度爭霸戰」與「正面交鋒–創意做法擂台戰」。「線上對決–模型準度爭霸戰」為2019/09/06 – 2019/11/22於T-Brain平台上傳預測結果的競賽，將以預測準確度為排名依據，爭取最高12萬元的獎金；「正面交鋒–創意做法擂台戰」將於頒獎典禮當天舉行，獲獎資格為曾於「線上對決–模型準度爭霸戰」上傳成功且「非前六名」得獎的參賽者，每一獎項皆有新台幣5,000(含)元以上獎金，獲獎機率高達20%，只要您願意分享您的建模做法，就有機會將獎項抱回家！

Rolling Mode Feature & Leakage Mode
==================
* **Insight:**
信用卡有特定通路使用、無腦刷、適合外國消費的雙幣卡等各種功能，其中特定通路使用的卡片在MCC、交易類型等Feature，眾數的集中程度應該是非常顯著的；若特定通路卡片之眾數有所改變，可以視為消費習性與個人過去消費習性相比有所改變。

* **Problem:**
若直接Groupby卡號計算眾數並視為Feature，此時會有把未來資料提前納入考量的leakage問題。

* **Solution:**
計算單一卡號在此筆消費以前的眾數為何，若有兩個眾數再透過Grouping leakage mode去修正。

說明
=================================================================================================
![image](https://github.com/CubatLin/TBrain-E.SUN-AI-Open-Competition-Fall-2019-15th-place-Feature-Engineering/blob/master/Mode_Demonstration.jpg)

* **df:**
  * Time:模擬交易時間
  * GroupKey:模擬卡號
  * Cat1:模擬MCC、交易類別等類別變數
  * ID:模擬JoinKey(txkey)

* **Cat1_ModeFrame:**
  * ID:模擬JoinKey(txkey)
  * GroupKey:模擬卡號
  * Cat1_Mode:Group Rolling Mode
  * Cat1_Mode_count:有幾個Mode
  * Cat1_Mode_Leckage: Group Leakage Mode
