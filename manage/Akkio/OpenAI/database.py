import sqlite3
import pandas as pd
import os


class SQLiteDB:
    def __init__(self):
        pass

    def table_creation(self):
        """
            Method Name: table_creation
            Description: This method is used to create a new table in database .
            Output:  Returns html table created from input data.
            On Failure: None
            Written By: Digiotai
            Version: 1.0
            Revisions: None
        """

        try:
            with sqlite3.connect("db.sqlite3") as c:
                # Getting the uploaded file
                cursor = c.cursor()
                query = """
                CREATE TABLE user_tracking (
                  user_id INT PRIMARY KEY,
                  Quota VARCHAR(40) NOT NULL DEFAULT FREE,
                  count INT DEFAULT 5,
                  email varchar(50)
                  );"""
                cursor.execute(query)
                c.commit()
        except Exception as e:
            print(e)

    def table_deletion(self):
        """
            Method Name: tabledeletion
            Description: This method is used to delete an existing table from database .
            Output:  None
            On Failure: None

            Written By: Digiotai
            Version: 1.0
            Revisions: None
        """
        try:
            with sqlite3.connect("db.sqlite3") as c:
                # Getting the uploaded file
                cursor = c.cursor()
                query = """
                DROP TABLE {name}
                  );"""
                cursor.execute(query)
                c.commit()
        except Exception as e:
            print(e)

    def add_user(self, user_id, email):
        try:
            with sqlite3.connect("db.sqlite3") as c:
                # Getting the uploaded file
                cursor = c.cursor()
                query = f"""
                INSERT INTO user_tracking (user_id, email) VALUES('{user_id}','{email}');"""
                cursor.execute(query)
                c.commit()
        except Exception as e:
            print(e)

    def get_user_data(self, user_id):
        try:
            with sqlite3.connect("db.sqlite3") as c:
                # Getting the uploaded file
                cursor = c.cursor()
                query = f"""SELECT * from user_tracking where user_id='{user_id}'"""
                cursor.execute(query)
                res = cursor.fetchone()
                print(res)
                c.commit()
                return res
        except Exception as e:
            print(e)

    def update_count(self, user_id):
        try:
            with sqlite3.connect("db.sqlite3") as c:
                # Getting the uploaded file
                cursor = c.cursor()
                query = f"""UPDATE user_tracking SET count = count - 1 where user_id='{user_id}'"""
                cursor.execute(query)
                res = cursor.fetchone()
                c.commit()
                return res
        except Exception as e:
            print(e)

    def update_user(self, user_id, plan, count):
        with sqlite3.connect("db.sqlite3") as c:
            # Getting the uploaded file
            cursor = c.cursor()
            query = f"""UPDATE user_tracking SET count = count - 1,Quota='{plan}',count='{count}' where user_id='{user_id}'"""
            cursor.execute(query)
            res = cursor.fetchone()
            c.commit()
            return res

    def get_users(self):
        try:
            with sqlite3.connect("db.sqlite3") as c:
                # Getting the uploaded file
                cursor = c.cursor()
                query = f"""SELECT user_id from user_tracking"""
                cursor.execute(query)
                res = cursor.fetchall()
                res = [i[0] for i in res]
                c.commit()
                return res
        except Exception as e:
            print(e)



if __name__ == "__main__":
    db = SQLiteDB()
    # db.table_creation()
    db.add_user('kpi')
    print(db.get_user_data('test'))
    # print(db.update_count('Rami'))
    # print(db.get_user_data('Rami'))





# try:
#     with sqlite3.connect("db.sqlite3") as c:
#         cursor = c.cursor()
        
#         # Create a new table with the desired default value
#         query_create_new_table = """
#         CREATE TABLE user_tracking_new (
#           user_id INT PRIMARY KEY,
#           Quota VARCHAR(40) NOT NULL DEFAULT 'FREE',
#           count INT DEFAULT 5,
#           email VARCHAR(50)
#         );"""
#         cursor.execute(query_create_new_table)
        
#         # Copy data from the old table to the new table
#         query_copy_data = """
#         INSERT INTO user_tracking_new (user_id, Quota, count, email)
#         SELECT user_id, Quota, count, email FROM user_tracking;"""
#         cursor.execute(query_copy_data)
        
#         # Drop the old table
#         query_drop_old_table = "DROP TABLE user_tracking;"
#         cursor.execute(query_drop_old_table)
        
#         # Rename the new table to the old table's name
#         query_rename_table = "ALTER TABLE user_tracking_new RENAME TO user_tracking;"
#         cursor.execute(query_rename_table)
        
#         c.commit()
# except Exception as e:
#     print(e)
