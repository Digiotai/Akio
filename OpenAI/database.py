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

    def create_connection(self, user, password, database, host, port=5432):
        try:
            self.connection = psycopg2.connect(
                database=database,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.connection.autocommit = True
            print("Database connection established.")
            return self.get_tables_info()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.connection = None
            return "Connection failed"

    def ensure_connection(self):
        """
        Ensure the connection is active. Reconnect if necessary.
        """
        try:
            if self.connection is None or self.connection.closed:
                print("Reconnecting to the database...")
                self.create_connection(
                    user=PGUSER,
                    password=PGPASSWORD,
                    database=PGDATABASE,
                    host=PGHOST
                )
        except Exception as e:
            print(f"Error ensuring connection: {e}")
            raise

    def create_table(self):
        """
        Create a table with a composite unique constraint on email and name.
        """
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                query = """CREATE TABLE IF NOT EXISTS akio_data(
                            id SERIAL PRIMARY KEY,
                            email VARCHAR(255),
                            name VARCHAR(255),
                            lastupdate TIMESTAMP,
                            datecreated TIMESTAMP,
                            fileobj BYTEA,
                            CONSTRAINT email_name_unique UNIQUE(email, name) -- Composite unique constraint
                            )"""
                cursor.execute(query)
                print("Table 'akio_data' created successfully.")
        except Exception as err:
            print(f"Error creating table: {err}")
            return str(err)

    def insert_or_update(self, email, data, tb_name):
        """
        Insert or update records based on a combination of email and name.
        """
        try:
            self.ensure_connection()
            tb_name_clean = tb_name.split('.')[0]  # Strip extension
            blob_data = pickle.dumps(data)  # Serialize the data

            with self.connection.cursor() as cursor:
                # Check if a record with the same email and name exists
                cursor.execute("SELECT id FROM akio_data WHERE email = %s AND name = %s", (email, tb_name_clean))
                existing_record = cursor.fetchone()

                if existing_record:
                    # Update the existing record
                    cursor.execute("""UPDATE akio_data
                                      SET lastupdate = %s, fileobj = %s
                                      WHERE email = %s AND name = %s""",
                                   (datetime.now(), psycopg2.Binary(blob_data), email, tb_name_clean))
                    print(f"Record with email '{email}' and name '{tb_name_clean}' updated successfully.")
                    return "Record updated successfully"
                else:
                    # Insert a new record
                    cursor.execute("""INSERT INTO akio_data (email, name, lastupdate, datecreated, fileobj)
                                      VALUES (%s, %s, %s, %s, %s)""",
                                   (email, tb_name_clean, datetime.now(), datetime.now(), psycopg2.Binary(blob_data)))
                    print(f"New record with email '{email}' and name '{tb_name_clean}' inserted successfully.")
                    return "Record inserted successfully"
        except Exception as err:
            print(f"Error inserting or updating record: {err}")
            return str(err)

    def read(self):
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM akio_data")
                rows = cursor.fetchall()

            # Convert the result to a DataFrame
            columns = ['id', 'email', 'name', 'lastupdate', 'datecreated', 'fileobj']
            df = pd.DataFrame(rows, columns=columns)
            return df
        except Exception as err:
            print(f"Error reading data: {err}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error

    def get_tables_info(self):
        try:
            df = self.read()
            return df.iloc[:, :-1].to_json() if not df.empty else {}
        except Exception as err:
            print(f"Error getting table info: {err}")
            return {}

    def get_user_tables(self, user):
        try:
            df = self.read()
            return list(df[df['email'] == user]['name']) if not df.empty else []
        except Exception as err:
            print(f"Error getting user tables: {err}")
            return {}

    def get_table_data(self, table_name):
        try:
            df = self.read()
            data_row = df[(df['name'] == table_name)]["fileobj"]
            if data_row.empty:
                raise ValueError(f"No data found for table: {table_name}")
            data = pickle.loads(data_row.values[0].tobytes())
            return data
        except Exception as err:
            print(f"Error getting table data: {err}")
            return pd.DataFrame()

    def delete_table_data(self, email, table_name):
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM akio_data WHERE email = %s AND name = %s", (email, table_name))
                if cursor.rowcount == 0:
                    return f"No data found for email: {email} and table: {table_name}"
            return f"Record deleted successfully for email: {email} and table: {table_name}"
        except Exception as err:
            print(f"Error deleting table data: {err}")
            return str(err)


# Database configuration
PGHOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech'
PGDATABASE = 'test'
PGUSER = 'test_owner'
PGPASSWORD = 'tcWI7unQ6REA'

if __name__ == '__main__':
    pdd = PostgresDatabase()
    pdd.create_connection(PGUSER, PGPASSWORD, PGDATABASE, PGHOST)
    pdd.create_table()

    # # Test case: Users uploading files
    # df1 = pd.DataFrame({"Col1": [1, 2], "Col2": ["A", "B"]})
    # df2 = pd.DataFrame({"Col1": [3, 4], "Col2": ["C", "D"]})
    #
    # # User 1 uploads two files
    # print(pdd.insert_or_update('user1@gmail.com', df1, "SharedData"))
    # print(pdd.insert_or_update('user1@gmail.com', df2, "UniqueData"))
    #
    # # User 2 uploads the same file as User 1
    # print(pdd.insert_or_update('user2@gmail.com', df1, "SharedData"))

    # Fetch table info to verify the result
    print(pdd.get_tables_info())


#
# from pymongo import MongoClient
# from bson.binary import Binary
# from datetime import datetime
# import pandas as pd
# import pickle
#
#
# class MongoDBDatabase:
#     """
#     A class to manage MongoDB operations such as creating collections,
#     inserting, reading, deleting, and truncating data.
#     """
#
#     def __init__(self, uri="mongodb+srv://digiotai:digiotai%40123@cluster0.whrxp.mongodb.net/digiotai?retryWrites=true&w=majority&appName=Cluster0", database_name="digiotai"):
#         """
#         Initializes the MongoDB client and sets the target database and collection.
#         :param uri: MongoDB connection URI
#         :param database_name: Name of the target database
#         """
#         self.client = MongoClient(uri)
#         self.database = self.client[database_name]
#         self.collection = self.database["data"]  # Default collection named 'data'
#
#     def create_connection(self):
#         """
#         Ensures the connection to the database and checks if the 'data' collection is accessible.
#         :return: JSON representation of collection documents or a connection failure message.
#         """
#         try:
#             return self.get_documents_info()
#         except Exception as e:
#             print(e)
#             return "Connection failed"
#
#     def create_collection(self):
#         """
#         Placeholder function. MongoDB auto-creates collections on inserting data.
#         Prints a message to indicate that the 'data' collection will be created automatically.
#         """
#         try:
#             print("Collection 'data' created (if it didn't exist).")
#         except Exception as err:
#             print(err)
#
#     def insert(self, data, tb_name):
#         """
#         Inserts a record into the 'data' collection. If a record with the same name exists,
#         it deletes the existing record before inserting the new one.
#         :param data: Data to insert (will be serialized as binary)
#         :param tb_name: Name of the table or file associated with the data
#         :return: Success or error message
#         """
#         try:
#             tb_name_clean = tb_name.split('.')[0]  # Strip file extension
#             blob_data = Binary(pickle.dumps(data))  # Serialize the data to binary
#
#             # Check if the document already exists
#             existing_document = self.collection.find_one({"name": tb_name_clean})
#             if existing_document:
#                 self.collection.delete_one({"name": tb_name_clean})  # Delete existing record
#
#             # Insert the new record
#             self.collection.insert_one({
#                 "name": tb_name_clean,
#                 "lastupdate": datetime.now(),
#                 "datecreated": datetime.now(),
#                 "fileobj": blob_data
#             })
#             return "Record inserted/updated successfully"
#
#         except Exception as err:
#             print(err)
#             return str(err)
#
#     def read(self):
#         """
#         Reads all documents from the 'data' collection and converts them to a pandas DataFrame.
#         :return: DataFrame containing the collection data or an empty DataFrame if no data exists.
#         """
#         try:
#             documents = list(self.collection.find())  # Fetch all documents
#             if not documents:
#                 return pd.DataFrame()  # Return empty DataFrame if no documents
#
#             df = pd.DataFrame(documents)  # Convert to DataFrame
#             return df.drop("_id", axis=1)  # Exclude the MongoDB `_id` field
#         except Exception as err:
#             print(err)
#             return pd.DataFrame()
#
#     def get_documents_info(self):
#         """
#         Retrieves the metadata (excluding binary data) for all documents in the 'data' collection.
#         :return: JSON representation of document metadata or an empty dictionary if no data exists.
#         """
#         try:
#             df = self.read()
#             return df.to_json() if not df.empty else {}
#         except Exception as err:
#             print(err)
#             return {}
#
#     def get_document_data(self, document_name):
#         """
#         Fetches and deserializes the binary data for a specific document by name.
#         :param document_name: Name of the document to fetch
#         :return: Deserialized data or an empty DataFrame if not found
#         """
#         try:
#             document = self.collection.find_one({"name": document_name})
#             if not document:
#                 raise ValueError(f"No data found for document: {document_name}")
#             data = pickle.loads(document["fileobj"])  # Deserialize binary data
#             return data
#         except Exception as err:
#             print(err)
#             return pd.DataFrame()
#
#
#
#
#     def delete_document(self, document_name):
#         """
#         Deletes a specific document by name from the 'data' collection.
#         :param document_name: Name of the document to delete
#         :return: Success or failure message
#         """
#         try:
#             result = self.collection.delete_one({"name": document_name})
#             if result.deleted_count == 0:
#                 return f"No data found for document: {document_name}"
#             return f"Document deleted successfully for: {document_name}"
#         except Exception as err:
#             print(err)
#             return str(err)
#
#     def drop_collection(self):
#         """
#         Drops the 'data' collection, removing all documents and the collection itself.
#         :return: Success or error message
#         """
#         try:
#             self.collection.drop()
#             return "Collection dropped."
#         except Exception as err:
#             print(err)
#             return str(err)
#
#     def truncate_collection(self):
#         """
#         Deletes all documents from the 'data' collection without dropping the collection.
#         :return: Success or error message
#         """
#         try:
#             self.collection.delete_many({})  # Delete all documents
#             return "All data truncated from the 'data' collection successfully."
#         except Exception as err:
#             print(err)
#             return str(err)
#
#
#
#
# if __name__ == '__main__':
#     mdb = MongoDBDatabase()  # Initialize MongoDBDatabase instance
#     # Example usage:
#     # mdb.create_collection()
#
#     # df = pd.read_csv(r"path_to_your_file.csv")
#     # print(mdb.insert(df, "your_file.csv"))
#
#     #print(mdb.get_documents_info())
#     print(mdb.read())
#     # mdb.drop_collection()
#     # mdb.delete_document('your_file')
#     # print(mdb.truncate_collection())
