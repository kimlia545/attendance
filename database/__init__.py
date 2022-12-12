import pymysql as my

def selectLogin(uid,upw):
    conn=None
    row=None #에러방지
    try:
        # 1. 연결
        conn = my.connect(host='localhost',
            port=3306, 
            user='root',
            password='12341234',
            db='abscheck',
            charset='utf8',
            cursorclass=my.cursors.DictCursor 
        )
        with conn.cursor() as cs:
            sql = '''
                SELECT 
                    uid 
                FROM 
                    admin 
                WHERE
                    uid = %s 
                AND
                    upw = %s;
            '''
            #쿼리 수행
            cs.execute( sql, (uid,upw) ) 
            row = cs.fetchone() 
    except Exception as e:
        # 로그 처리
            print(e)
    finally:
            # 3. 연결 종료 : I/O
            if conn : #conn이 존재할때만
                conn.close() # 커서 닫기
            return row    