import pymysql
from database import connection
# import connection

def call_list():
    try:
        conn = connection.get_connetion()

        sql = '''
                SELECT * FROM member_list;
            '''

        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # print('디비줄',rows)

    finally:
        conn.close()

    return rows

def register_student(name, my_class, birdate, gender, abs_rate):
    result = 0
    try:
        conn = connection.get_connetion()
        sql = ''' 
                INSERT INTO member_list(name, class, birdate, gender, abs_rate)
                VALUES (%s, %s, %s, %s, %s);   
                                
                             
            '''
        with conn.cursor() as cursor:
                cursor.execute(sql, (name, my_class, birdate, gender, abs_rate))
            # 디비에 반영
        conn.commit()        
            # 검증    
        result = conn.affected_rows()

        # if result:# 입력되었다
        #     sql = '''
        #         SELECT * FROM member_list;
        #     ''' 
        #     with conn.cursor() as cursor:
        #         cursor.execute(sql)
        #         rows = cursor.fetchall()           
        # else:     # 입력 실패        
        #     pass
    except Exception as e:        
        print('에러', e)
    finally:
        if conn:
            conn.close()    
    
    return result

def get_name( name ):
    rows = None    
    conn = None
    try:
        conn = connection.get_connetion()    
        sql = '''
            INSERT INTO abs_log(m_id, isin, regdate)
            VALUES ((SELECT NO FROM member_list WHERE NAME=%s), 0, NOW());
            '''
        with conn.cursor() as cursor:
            cursor.execute(sql, (name))
        # 디비에 반영
        conn.commit()        
        # 검증    
        result = conn.affected_rows()

        if result:# 입력되었다
            sql = '''
                SELECT * FROM abs_log;
            ''' 
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
            pass
        else:     # 입력 실패        
            pass
    except Exception as e:        
        print('에러', e)
    finally:
        if conn:
            conn.close()    
    
    return rows    

def call_name():
    try:
        conn = connection.get_connetion()

        sql = '''
                SELECT 
                    row_number() OVER( ORDER BY A.regdate ) AS RowNum,
                    A.*, B.name, B.class from 
                (
                SELECT * FROM abs_log WHERE  DATE(NOW()) = DATE(regdate)
                ) AS A
                INNER JOIN member_list AS B
                ON A.m_id=B.no;
            '''

        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # print('디비줄',rows)

    finally:
        conn.close()

    return rows




if __name__=='__main__':
   
    pass
