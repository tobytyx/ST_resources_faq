# -*- coding: utf-8 -*-
import pymysql
import traceback
from pymysql.connections import Connection
MYSQL_USER = 'tyx'
MYSQL_PASS = 'tyx'
MYSQL_DB = 'faq_management'

STATE_ERROR_NUMBER = -1
STATE_READY_NUMBER = 0
STATE_TRAINING_NUMBER = 1
STATE_USING_NUMBER = 2

conn = None


def get_mysql_connect():
    global conn
    if conn is not None:
        return conn
    try:
        conn = pymysql.connect(
            host='localhost',
            port=3306,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=MYSQL_DB,
            charset='utf8mb4'
        )
    except Exception:
        traceback.print_exc()
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
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return record_id

def insert_model_record_new(conn: Connection, name, domain, state, categories, comment):
    try:
        name = str(name)
        domain = str(domain)
        state = str(state)
        category_num = len(categories)
        categories = ",".join([str(i) for i in categories])
        comment = str(comment)
        cur = conn.cursor()
        cur.execute(
            'insert into `faq_management_model`(`name`, `domain`, `state`, `categories`, `category_num`, `comment`) values (%s, %s, %s, %s, %s, %s)',
            args=(name, domain, state, categories, category_num, comment)
        )
        cur.execute(
            'SELECT LAST_INSERT_ID() from faq_management_model'
        )
        record_id = int(cur.fetchone()[0])
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return record_id


def delete_model_record(conn: Connection, record_id):
    try:
        record_id = str(record_id)
        cur = conn.cursor()
        record_num = cur.execute(
            'select state from faq_management_model where record_id=%s',
            args=record_id
        )
        if record_num == 0:
            return STATE_ERROR_NUMBER
        state = int(cur.fetchone()[0])
        if state == STATE_READY_NUMBER:
            cur.execute(
                'delete from `faq_management_model` where record_id=%s',
                args=record_id
            )
            conn.commit()
            cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return state


def update_model_record(conn: Connection, uid, state):
    try:
        uid = str(uid)
        state = str(state)
        cur = conn.cursor()
        update_num = cur.execute(
            'update faq_management_model set state=%s where record_id=%s',
            args=(state, uid)
        )
        if update_num == 0:
            return STATE_ERROR_NUMBER
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return 0


def init_model_record():
    try:
        conn = get_mysql_connect()
        cur = conn.cursor()
        cur.execute(
            'update faq_management_model set state=%s where state=%s',
            args=("0", "2")
        )
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return 0


def insert_category(conn: Connection, name, answer):
    try:
        name = str(name)
        answer = str(answer)
        cur = conn.cursor()
        cur.execute(
            'insert into `faq_management_category`(`name`, `answer`) values (%s, %s)',
            args=(name, answer)
        )
        cur.execute(
            'SELECT LAST_INSERT_ID() from faq_management_category'
        )
        category_id = int(cur.fetchone()[0])
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return category_id


def delete_category(conn: Connection, category_id):
    try:
        category_id = str(category_id)
        cur = conn.cursor()
        record_num = cur.execute(
            'select category_id from faq_management_category where category_id=%s',
            args=category_id
        )
        if record_num == 0:
            return STATE_ERROR_NUMBER
        cur.execute(
            'delete from `faq_management_category` where category_id=%s',
            args=category_id
        )
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return 0


def update_category(conn: Connection, category_id, answer):
    try:
        category_id = str(category_id)
        answer = str(answer)
        cur = conn.cursor()
        update_num = cur.execute(
            'update faq_management_category set answer=%s where category_id=%s',
            args=(answer, category_id)
        )
        if update_num == 0:
            return STATE_ERROR_NUMBER
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return 0


def get_category_queries(conn: Connection, category_id):
    cur = conn.cursor()
    query_num = cur.execute(
        'select text from faq_management_query where category_id=%s',
        args=(category_id)
    )
    if query_num == 0:
        return [], 0
    texts = [r[0] for r in cur.fetchall()]
    cur.close()
    return texts, query_num


def get_category_answer(conn: Connection, category_id):
    cur = conn.cursor()
    query_num = cur.execute(
        'select answer from faq_management_category where category_id=%s',
        args=(category_id)
    )
    if query_num == 0:
        return ""
    answer = cur.fetchone()[0]
    cur.close()
    return answer


def insert_query(conn: Connection, category_id, text):
    try:
        category_id = str(category_id)
        text = str(text)
        cur = conn.cursor()
        cur.execute(
            'insert into `faq_management_query`(`category_id`, `text`) values (%s, %s)',
            args=(category_id, text)
        )
        cur.execute(
            'SELECT LAST_INSERT_ID() from faq_management_query'
        )
        query_id = int(cur.fetchone()[0])
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return query_id


def delete_query(conn: Connection, query_id):
    try:
        query_id = str(query_id)
        cur = conn.cursor()
        record_num = cur.execute(
            'select category_id from faq_management_query where query_id=%s',
            args=query_id
        )
        if record_num == 0:
            return STATE_ERROR_NUMBER
        cur.execute(
            'delete from `faq_management_query` where query_id=%s',
            args=query_id
        )
        conn.commit()
        cur.close()
    except Exception:
        traceback.print_exc()
        return STATE_ERROR_NUMBER
    return 0
