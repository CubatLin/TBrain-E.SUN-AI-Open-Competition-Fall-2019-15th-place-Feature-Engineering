玉山銀行人工智慧公開挑戰賽_2019秋季賽_真相只有一個_信用卡盜刷偵測-15th place(top1%) Feature Engineering
======================================
* **比賽網址：**[TBrain E.SUN AI Open Competition Fall 2019 ](https://tbrain.trendmicro.com.tw/Competitions/Details/10)

* **參賽成績：**
```
Team:菜雞互啄
Members:Alexi,Pat,Michael,Ming-Xiang,Ethan
F1 score prediction competition: 15th of 1366 teams
Creativity presentation competition: 2nd of final 20 teams
```
* **參賽海報(閃電講2nd)：**[菜雞互啄_海報](https://github.com/CubatLin/TBrain-E.SUN-AI-Open-Competition-Fall-2019-15th-place-Feature-Engineering/blob/master/%E8%8F%9C%E9%9B%9E%E4%BA%92%E5%95%84_%E7%8E%89%E5%B1%B1%E9%8A%80%E8%A1%8C2019%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E7%A7%8B%E5%AD%A3%E5%85%AC%E9%96%8B%E6%8C%91%E6%88%B0%E8%B3%BD%E5%85%A8%E9%96%8B%E6%B5%B7%E5%A0%B1.pdf)
* **Feature Engineering：**[Python Code](https://github.com/CubatLin/TBrain-E.SUN-AI-Open-Competition-Fall-2019-15th-place-Feature-Engineering/blob/master/%E8%8F%9C%E9%9B%9E%E4%BA%92%E5%95%84_ModeCode.py)

* **菜雞成員Link：**[Alex Lau](https://github.com/alexlautw9527), [Pat Wu](https://github.com/ts01174755),[Michael Shen](https://github.com/changsian), [Ming-Xiang](https://github.com/kuo23)


Description 
===========
一卡在手，妙用無窮！ 在台灣，20歲以上持有信用卡人數超過六成。因信用卡具備高回饋、延遲付款以及付款便利等特性，使得信用卡成為人們支付時不可或缺的工具。不過隨著科技的日新月異，不肖分子也針對此支付模式衍生出新的犯罪手法，即「信用卡盜刷」。

面對盜刷，一般民眾除了可以透過經常對帳、防止卡片資訊外洩等方式來避免外，國內外銀行及發卡組織近年也開始運用機器學習演算法找出潛在的盜刷交易，並及早因應。然而，盜刷的樣態千百種，到底什麼才是足以判斷為盜刷的關鍵因子呢？

本次競賽提供去識別信用卡交易授權資料，希望大家集思廣益，一同「反盜刷」！不僅捍衛自己的資產，守護身邊親友的財富，更有機會獲得高額獎金！

本次競賽共包含兩場獨立賽事，分別為「線上對決–模型準度爭霸戰」與「正面交鋒–創意做法擂台戰」。「線上對決–模型準度爭霸戰」為2019/09/06 – 2019/11/22於T-Brain平台上傳預測結果的競賽，將以預測準確度為排名依據，爭取最高12萬元的獎金；「正面交鋒–創意做法擂台戰」將於頒獎典禮當天舉行，獲獎資格為曾於「線上對決–模型準度爭霸戰」上傳成功且「非前六名」得獎的參賽者，每一獎項皆有新台幣5,000(含)元以上獎金，獲獎機率高達20%，只要您願意分享您的建模做法，就有機會將獎項抱回家！


變數名稱與解釋
=================================================================================================
欄位          | 說明  |備註
-------------|:-----:|---------------------------------------------------------------------------------------------------
bacno | 歸戶帳號	|持卡人帳戶
txkey	|交易序號	|一筆交易產生一組交易序號(唯一值)
locdt	|授權日期	|刷卡交易時向發卡銀行提出授權之日期(通常與交易日期為同一天，除非有預先授權的情況，參考：https://cpsamuelsl527.pixnet.net/blog/post/63542071-[教學]-「預先授權」到底是什麼？)
loctm	|授權時間	|刷卡交易時向發卡銀行提出授權之時間
cano	|交易卡號	|交易卡號(一個持卡人帳戶下可以有很多組卡號)
contp	|交易類別	|交易類型，舉例：正向交易、負向交易(刷退)、預借現金等
etymd	|交易型態	|交易方式，舉例：手動輸入卡號、刷磁條機、感應讀取卡號、利用預先儲存的信用卡資料進行交易(例如Netflix月租自動扣款) 參考資料：https://dnwebdomestic2.efunds.com/dnweb/webhelp/Field_Descriptions/MC_POS_Entry_Mode.htm
mchno	|特店代號	|消費商店之代碼
acqic	|收單行代碼	|消費商店合作銀行的代碼
mcc	|MCC_CODE	|消費子類別代碼(一個消費類別(例如保險、航空、食品等)下面可以有很多MCC_CODE)
conam	|交易金額-台幣(經過轉換)	|  
ecfg	|網路交易註記	|交易是否在網路上完成
insfg	|分期交易註記	|交易是否有分期
iterm	|分期期數	|
stocn	|消費地國別	|消費商店登記在哪個國家
scity	|消費城市	|消費商店登記在哪個城市
stscd	|狀態碼	|卡片狀態，舉例：停卡、掛失、逾期等
ovrlt	|超額註記碼	|累積使用額度超過額度上限，即被下註記
flbmk	|Fallback 註記	|採用人工授權方式所下的註記，其可能原因為隨機抽樣或是發卡銀行懷疑此筆為不正常交易
hcefg	|支付形態	|虛擬卡交易註記，其型態可能為APPLE PAY, android pay, Samsong pay 等
csmcu	|消費地幣別	|使用哪個幣別進行消費
flg_3dsmk	|3DS 交易註記	|有進行手機驗證碼之交易
fraud_ind	|盜刷註記	|是否有盜刷(應變數)



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
  * Cat1_Mode_Leakage: Group Leakage Mode
