import pymysql

def get_connetion():
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='12341234',
                           db='abscheck', charset='utf8', cursorclass=pymysql.cursors.DictCursor)
    return conn