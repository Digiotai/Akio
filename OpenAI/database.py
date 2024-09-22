# import psycopg2
# class PostgreSQLDB:
#     def __init__(self, dbname, user, password, host='cornelius.db.elephantsql.com', port=5432):
#         self.dbname = dbname
#         self.user = user
#         self.password = password
#         self.host = host
#         self.port = port
#
#     def connect(self):
#         try:
#             conn = psycopg2.connect(
#                 dbname=self.dbname,
#                 user=self.user,
#                 password=self.password,
#                 host=self.host,
#                 port=self.port
#             )
#             return conn
#         except Exception as e:
#             print(e)
#             return None
#
#     def table_creation(self):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = """
#                 CREATE TABLE IF NOT EXISTS user_tracking (
#                     user_name VARCHAR(50) PRIMARY KEY,
#                     quota VARCHAR(40) NOT NULL DEFAULT 'FREE',
#                     count INT DEFAULT 5,
#                     email VARCHAR(50) UNIQUE
#                 );
#                 """
#                 cursor.execute(query)
#                 conn.commit()
#                 cursor.close()
#                 conn.close()
#         except Exception as e:
#             print(e)
#
#     def table_deletion(self):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = "DROP TABLE IF EXISTS user_tracking;"
#                 cursor.execute(query)
#                 conn.commit()
#                 cursor.close()
#                 conn.close()
#         except Exception as e:
#             print(e)
#
#     def add_user(self, user_name, email):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = """
#                 INSERT INTO user_tracking (user_name, email)
#                 VALUES(%s, %s);
#                 """
#                 cursor.execute(query, (user_name, email))
#                 conn.commit()
#                 cursor.close()
#                 conn.close()
#         except Exception as e:
#             print(e)
#
#     def get_user_data(self, user_name):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = "SELECT * FROM user_tracking WHERE user_name=%s;"
#                 cursor.execute(query, (user_name,))
#                 res = cursor.fetchone()
#                 cursor.close()
#                 conn.close()
#                 return res
#         except Exception as e:
#             print(e)
#             return None
#
#     def update_count(self, user_name):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = "UPDATE user_tracking SET count = count - 1 WHERE user_name=%s;"
#                 cursor.execute(query, (user_name,))
#                 conn.commit()
#                 cursor.close()
#                 conn.close()
#         except Exception as e:
#             print(e)
#
#     def update_user(self, user_name, plan, count):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = """
#                 UPDATE user_tracking
#                 SET quota=%s, count=%s
#                 WHERE user_name=%s;
#                 """
#                 cursor.execute(query, (plan, count, user_name))
#                 conn.commit()
#                 cursor.close()
#                 conn.close()
#         except Exception as e:
#             print(e)
#
#     def get_users(self):
#         try:
#             conn = self.connect()
#             if conn is not None:
#                 cursor = conn.cursor()
#                 query = "SELECT user_name FROM user_tracking;"
#                 cursor.execute(query)
#                 res = cursor.fetchall()
#                 res = [i[0] for i in res]
#                 cursor.close()
#                 conn.close()
#                 return res
#         except Exception as e:
#             print(e)
#             return None
#
#
# if __name__ == "__main__":
#     db = PostgreSQLDB(dbname='uibmogli', user='uibmogli', password='8ogImHfL_1G249lXtM3k2EAIWTRDH2mX')
#     db.table_creation()
#     db.add_user("revirevi", 'test@example123.com')
#     print(db.get_user_data('revirevi'))


import psycopg2
#
# # For storage of tables and all
# import psycopg2
# import pickle
# import pandas as pd
# import psycopg2
# from datetime import datetime
# import psycopg2.extras
#
# class PostgresDatabase:
#     def __init__(self):
#         self.connection = None
#
#     def create_connection(self, user, password, database, host, port):
#         try:
#             self.connection = psycopg2.connect(
#                 database=database,
#                 user=user,
#                 password=password,
#                 host=host,
#                 port=port
#             )
#             return self.get_tables_info()
#         except Exception as e:
#             print(e)
#             return "connection failed"
#
#     def get_tables(self):
#         try:
#             cursor = self.connection.cursor()
#             cursor.execute("""SELECT table_name
#                               FROM information_schema.tables
#                               WHERE table_schema = 'public'
#                               AND table_type = 'BASE TABLE';""")
#             table_names = [d[0] for d in cursor.fetchall()]
#             return table_names
#         except Exception as e:
#             print(e)
#
#     def create_table(self):
#         cursor = self.connection.cursor()
#         cursor.execute(f"DROP TABLE IF EXISTS data;")
#         query = f"""CREATE TABLE data(
#                         id SERIAL PRIMARY KEY,
#                         name VARCHAR(255),
#                         lastupdate TIMESTAMP,
#                         datecreated TIMESTAMP,
#                         fileobj BYTEA)"""
#         try:
#             cursor.execute(query)
#             self.connection.commit()
#             cursor.close()
#         except Exception as err:
#             print(err)
#             cursor.close()
#             return str(err)
#
#     def insert(self, data, tb_name):
#         cursor = self.connection.cursor()
#         try:
#             tb_name_clean = tb_name.split('.')[0]  # Strip extension
#             blob_data = psycopg2.Binary(pickle.dumps(data))
#             query = f"""INSERT INTO data (name, lastupdate, datecreated, fileobj)
#                         VALUES (%s, %s, %s, %s)"""
#             cursor.execute(query, (tb_name_clean, datetime.now(), datetime.now(), blob_data))
#             self.connection.commit()
#             cursor.close()
#             return "Records inserted successfully!"
#         except Exception as err:
#             print(err)
#             cursor.close()
#             return err
#
#     def read(self):
#         try:
#             query = "SELECT * FROM data"
#             df = pd.read_sql_query(query, self.connection)
#             return df
#         except Exception as err:
#             print(err)
#             return pd.DataFrame()  # Return an empty DataFrame in case of error
#
#     def get_tables_info(self):
#         try:
#             df = self.read()
#             return df.iloc[:, :-1].to_json() if not df.empty else {}
#         except Exception as err:
#             print(err)
#             return {}
#
#     def get_table_data(self, table_name):
#         try:
#             df = self.read()
#             data_row = df[df['name'] == table_name]["fileobj"]
#             if data_row.empty:
#                 raise ValueError(f"No data found for table: {table_name}")
#             data = pickle.loads(data_row.values[0].tobytes())
#             return data
#         except Exception as err:
#             print(err)
#             return pd.DataFrame()
#
# if __name__ == '__main__':
#     pdd = PostgresDatabase()
#     pdd.create_connection("username", "password", "dbname", "host", "5432")
#     # pdd.create_table()
#     # df = pd.read_csv(r"C:\path_to_your_file\retail_sales_data.csv")
#     # print(pdd.insert(df, "retail_sales_data.csv"))
#     # print(pdd.get_tables_info())
#     # print(pdd.get_table_data('retail_sales_data'))


import psycopg2
import pandas as pd
import pickle
from datetime import datetime


class PostgresDatabase:
    def __init__(self):
        self.connection = None
        self.headers = {
            'Authorization':
                'FlespiToken axLBthbazeJkKKkpr2sVK9rAeXfFJGmH1V9k18iqaSyKqHYHzetadIyitBL15WyU'
        }
        self.data = None

    def create_connection(self, user, password, database, host, port):
        try:
            self.connection = psycopg2.connect(
                database=database,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.connection.autocommit=True
            return self.get_tables_info()
        except Exception as e:
            print(e)
            return "connection failed"

    def get_tables(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""SELECT table_name 
                                  FROM information_schema.tables 
                                  WHERE table_schema = 'public'  
                                  AND table_type = 'BASE TABLE';""")
                table_names = [row[0] for row in cursor.fetchall()]
            return table_names
        except Exception as e:
            print(e)

    def create_table(self):
        try:
            with self.connection.cursor() as cursor:
                query = """CREATE TABLE IF NOT EXISTS data(
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(255) UNIQUE,  -- Ensure 'name' is unique
                            lastupdate TIMESTAMP,
                            datecreated TIMESTAMP,
                            fileobj BYTEA)"""
                cursor.execute(query)
                print("Table 'data' created successfully.")
        except Exception as err:
            print(err)
            return str(err)

    def insert(self, data, tb_name):
        try:
            tb_name_clean = tb_name.split('.')[0]  # Strip extension
            blob_data = pickle.dumps(data)  # Serialize the data

            with self.connection.cursor() as cursor:
                # Check if the entry exists
                cursor.execute("SELECT COUNT(*) FROM data WHERE name = %s", (tb_name_clean,))
                if cursor.fetchone()[0] > 0:
                    cursor.execute("DELETE FROM data WHERE name = %s", (tb_name_clean,))

                # Insert the new data
                cursor.execute("""INSERT INTO data (name, lastupdate, datecreated, fileobj) 
                                  VALUES (%s, %s, %s, %s)""",
                               (tb_name_clean, datetime.now(), datetime.now(), psycopg2.Binary(blob_data)))
            return "Record inserted/updated successfully"

        except Exception as err:
            print(err)
            return str(err)

    def read(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM data")
                rows = cursor.fetchall()

            # Convert the result to a DataFrame
            columns = ['id', 'name', 'lastupdate', 'datecreated', 'fileobj']
            df = pd.DataFrame(rows, columns=columns)
            return df

        except Exception as err:
            print(err)
            return pd.DataFrame()  # Return an empty DataFrame in case of error

    def get_tables_info(self):
        try:
            df = self.read()
            return df.iloc[:, :-1].to_json() if not df.empty else {}
        except Exception as err:
            print(err)
            return {}

    def get_table_data(self, table_name):
        try:
            df = self.read()
            data_row = df[df['name'] == table_name]["fileobj"]
            if data_row.empty:
                raise ValueError(f"No data found for table: {table_name}")
            data = pickle.loads(data_row.values[0].tobytes())
            return data
        except Exception as err:
            print(err)
            return pd.DataFrame()

    def delete_table_data(self, table_name):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM data WHERE name = %s", (table_name,))
                if cursor.rowcount == 0:
                    return f"No data found for table: {table_name}"
            return f"Record deleted successfully for table: {table_name}"
        except Exception as err:
            print(err)
            return str(err)

    def drop_table(self, table_name):
        """Drop a specific table from the database."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            return f"Table '{table_name}' dropped successfully."
        except Exception as err:
            print(err)
            return str(err)

    def truncate_data_table(self):
        """Truncate the data table to delete all rows but keep the structure intact."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE data CASCADE")
            return "All data truncated from the 'data' table successfully."
        except Exception as err:
            print(err)
            return str(err)


if __name__ == '__main__':
    pdd = PostgresDatabase()  # Uses default credentials
    # Example usage:
    #pdd.create_table()

    # df = pd.read_csv(r"path_to_your_file.csv")
    # print(pdd.insert(df, "your_file.csv"))

    print(pdd.get_tables_info())
    #pdd.drop_table("data")
    # pdd.delete_table_data('data')
    # Truncate all data in the 'data' table
    #print(pdd.truncate_data_table())
