# -*- coding: utf-8 -*-
import pymysql
from pymysql.connections import Connection
MYSQL_USER = 'tyx'
MYSQL_PASS = 'tyx'
MYSQL_DB = 'faq_management'
MYSQL_TABLE = 'faq_management_model'

STATE_ERROR_NUMBER = -1
STATE_READY_NUMBER = 0
STATE_TRAINING_NUMBER = 1
STATE_USING_NUMBER = 2


def get_mysql_connect():
    try:
        conn = pymysql.connect(
            host='localhost',
            port=3306,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=MYSQL_DB,
            charset='utf8mb4'
        )
    except:
        conn = None
    return conn


def insert_model_record(conn: Connection, name, domain, state, data_path, category_num, comment):
    try:
        name = str(name)
        domain = str(domain)
        state = str(state)
        data_path = str(data_path)
        category_num = str(category_num)
        comment = str(comment)
        cur = conn.cursor()
        cur.execute(
            'insert into `faq_management_model`(`name`, `domain`, `state`, `data_path`, `category_num`, `comment`) values (%s, %s, %s, %s, %s, %s)',
            args=(name, domain, state, data_path, category_num, comment)
        )
        cur.execute(
            'SELECT LAST_INSERT_ID() from faq_management_model'
        )
        record_id = int(cur.fetchone()[0])
        conn.commit()
        cur.close()
    except:
        return -1
    return record_id


def delete_model_record(conn: Connection, uid):
    try:
        uid = str(uid)
        cur = conn.cursor()
        record_num = cur.execute(
            'select state from faq_management_model where record_id=%s',
            args=uid
        )
        if record_num == 0:
            return STATE_ERROR_NUMBER
        state = int(cur.fetchone()[0])
        if state == STATE_READY_NUMBER:
            cur.execute(
                'delete from `faq_management_model` where record_id=%s',
                args=uid
            )
            conn.commit()
            cur.close()
    except:
        return STATE_ERROR_NUMBER
    return state


def update_model_record(conn: Connection, uid, state):
    try:
        uid = str(uid)
        state = str(uid)
        cur = conn.cursor()
        update_num = cur.execute(
            'update faq_management_model set state=%s where record_id=%s',
            args=(state, uid)
        )
        if update_num == 0:
            return -1
        conn.commit()
        cur.close()
    except:
        return -1
    return 0
