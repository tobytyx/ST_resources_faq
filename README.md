## 科技资源领域的问答系统

## 简介
本项目采用相似问匹配的方式完成问答任务，在预先添加一部分数据后，模型进行学习并将新出现的问题匹配到原有相似问中。

除了常规的模型训练，本项目还包括了自动训练、数据管理的后端功能，后端使用FastAPI轻框架。数据库使用MySQL，具体数据格式后续会进行介绍。

### 模型介绍
模型采用双塔结构，Base Model采用Bert，支持普通的双塔Bi-Bert以及Poly-Encoder结构。

### 数据介绍
数据为两层结构，第一层是标准问类别，例如“专利申请日期类型的问题”；第二层是每个标准问下的相似问，例如“专利申请日期是什么？”。

具体数据库中的保存结构详见```tables.sql```


## 项目结构
```console
├── auto_train.py
├── auto_train.sh
├── data
│   ├── equipment
│   │   ├── test.tsv
│   │   ├── total.tsv
│   │   └── train.tsv
│   ├── expert
│   │   ├── test.tsv
│   │   ├── total.tsv
│   │   └── train.tsv
│   └── patent
│       ├── test.tsv
│       ├── total.tsv
│       └── train.tsv
├── dataset.py
├── dependences
│   └── roberta_wwm_ext -> /home/tanyx/dependences/roberta_wwm_ext/
├── __init__.py
├── log.py
├── model.py
├── mysql_utils.py
├── output
├── README.md
├── run_server.sh
├── server.py
├── tables.sql
├── test.py
├── tools.py
└── train.py
```

## 运行方式
### 线下运行
```
python train.py  # 详细参数设置请参照train.py
```
### 线上运行
```
uvicorn server:app --port 9900 --host 0.0.0.0

# 上线模型
POST
IP:9900/update/
JSON:{
	"record_id": 1,
	"domain": "patent",
	"name": "bi_patent"
}

# 查询faq
GET
IP:9900/faq/?domain=patent&text=专利时间是啥
```