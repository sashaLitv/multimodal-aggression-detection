import subprocess
from pathlib import Path
import pyodbc
import docker
import time
from docker.errors import NotFound
import os
import re

class DatabaseManager:
    def __init__(self):
        self.docker_config = {
            "server": os.getenv("DB_SERVER", "127.0.0.1,1433"),
            "driver": os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}"),
            "image_name": os.getenv("DB_DOCKER_IMAGE", "mcr.microsoft.com/azure-sql-edge"),
            "container_name": os.getenv("DB_DOCKER_CONTAINER", "sql"),
        }

        self.db_config = {
            "database": os.getenv("DB_NAME", "aggression_monitoring"),
            "uid": os.getenv("DB_USER", "sa"),
            "pwd": os.getenv("DB_PASSWORD", "VeryStr0ngP@ssw0rd"),
        }

        self._query_cache = {}
        if not self._is_db_ready():
            self._ensure_docker_daemon_running()
            self._ensure_docker_container()

        self._init_db_schema()

        self.conn = self._get_connection(self.db_config["database"])

    def _ensure_docker_daemon_running(self):
        try:
            client = docker.from_env()
            client.ping()
            print("Docker Daemon активний.")
        except Exception:
            print("Docker Daemon не знайдено. Запуск Docker Desktop...")
            subprocess.run(["open", "-g", "-a", "Docker"], check=True)
            
            max_attempts = 20
            for i in range(max_attempts):
                try:
                    client = docker.from_env()
                    client.ping()
                    print("Docker Desktop успішно ініціалізовано.")
                    return
                except Exception:
                    print(f"Очікування завантаження Docker ({i+1}/{max_attempts})...")
                    time.sleep(5)
            raise RuntimeError("Не вдалося запустити Docker Desktop.")
    def _ensure_docker_container(self):
        client = docker.from_env()

        try:
            container = client.containers.get(self.docker_config["container_name"])
            if container.status != "running":
                print(f"Контейнер {self.docker_config["container_name"]} знайдено, але він зупинений. Запуск...")
                container.start()
            else:
                print(f"Контейнер {self.docker_config["container_name"]} вже працює.")

        except NotFound:
            container = client.containers.run(
                image = self.docker_config["image_name"],
                name = self.docker_config["container_name"],
                detach = True,
                environment = {            
                    'ACCEPT_EULA': 'Y',
                    'MSSQL_SA_PASSWORD': self.db_config["pwd"],
                },
                ports={'1433/tcp': 1433}, 
                cap_add = ["SYS_PTRACE"]
            )

        self._wait_for_sql_ready()

    def _is_db_ready(self):
        try:
            conn = self._get_connection()
            conn.close()
            return True
        except Exception:
            return False
    def _init_db_schema(self):
        file_order = [
            "users.sql",
            "data_source.sql",
            "incident.sql",
            "media_proof.sql",
            "audit.sql",
        ]
        db_folder = Path(__file__).resolve().parent / "dbo"
        if not db_folder.exists():
            raise FileNotFoundError(f"Не знайдено папку зі скриптами: {db_folder}")

        safe_db_name = self.db_config["database"].replace("]", "]]")
        safe_db_literal = self.db_config["database"].replace("'", "''")

        conn_master = self._get_connection("master")
        try:
            cursor_master = conn_master.cursor()
            cursor_master.execute(
                f"IF DB_ID(N'{safe_db_literal}') IS NULL CREATE DATABASE [{safe_db_name}];"
            )

            cursor_master.close()
        finally:
            conn_master.close()

        sql_files = [db_folder / name for name in file_order]

        conn_agression = self._get_connection(self.db_config["database"])
        try:
            cursor_agression = conn_agression.cursor()
            for file_path in sql_files:
                if not file_path.exists():
                    raise FileNotFoundError(f"SQL-файл не знайдено: {file_path}")

                table_name = file_path.stem 
                cursor_agression.execute(
                    "SELECT OBJECT_ID(?, N'U')", (f"dbo.{table_name}",)
                )
                if cursor_agression.fetchone()[0] is not None:
                    continue

                sql_text = file_path.read_text(encoding="utf-8")
                batches = re.split(
                    r"^\s*GO\s*$", sql_text, flags=re.MULTILINE | re.IGNORECASE
                )
                for batch in batches:
                    stmt = batch.strip()
                    if stmt:
                        cursor_agression.execute(stmt)

            cursor_agression.execute("SELECT COUNT(*) FROM [dbo].[users]")
            if cursor_agression.fetchone()[0] == 0:
                cursor_agression.execute("""
                    INSERT INTO [dbo].[users] (last_name, first_name, middle_name, profile_type)
                    VALUES
                        (N'Оператор', N'Тестовий', N'-', N'Оператор'),
                        (N'Аналітик', N'Тестовий', N'-', N'Аналітик');
                """)

            cursor_agression.close()
        finally:
            conn_agression.close()

        
    def _get_connection(self, database="master"):
        connection_string = f'DRIVER={self.docker_config["driver"]};SERVER={self.docker_config["server"]};DATABASE={database};UID={self.db_config["uid"]};PWD={self.db_config["pwd"]}'
        return pyodbc.connect(connection_string, autocommit=True)
    
    def _wait_for_sql_ready(self):
        max_retries = 15
        conn_str = f'DRIVER={self.docker_config["driver"]};SERVER={self.docker_config["server"]};UID={self.db_config["uid"]};PWD={self.db_config["pwd"]}'
        for i in range(max_retries):
            try:
                conn = pyodbc.connect(conn_str, timeout=2) 
                conn.close()
                return True
            except Exception:
                time.sleep(2)
    def stop_docker_container(self):
        try:
            client = docker.from_env()
            container = client.containers.get(self.docker_config["container_name"])

            if container.status == "running":
                print(f"Контейнер {self.docker_config["container_name"]} знайдено, зупинка..")
                container.stop()
        except NotFound:
            print(f"Контейнер {self.docker_config["container_name"]} не знайдено.")


    def _load_query(self, filename: str) -> str:
        if filename not in self._query_cache:
            path = Path(__file__).resolve().parent / "queries" / filename
            self._query_cache[filename] = path.read_text(encoding="utf-8")
        return self._query_cache[filename]

    def create_audit_record(self, id_user, id_source, start_time, end_time):
        cursor = self.conn.cursor()
        try:
            query = self._load_query("create_audit_record.sql")
            cursor.execute(query, (id_user, id_source, start_time, end_time)) # id_user, id_source, audit_start, audit_end
            
            
            row = cursor.fetchone()
            if row:
                id_new = int(row[0])
                print(f"Аудит збережено: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')} (ID: {id_new})")
                return id_new
        except Exception as e:
            print(f"Audit Error: {e}")
    def process_incident_segment(self, id_source, id_user, media_type, current_time, status, confidence, media_path):
        if confidence is not None:
            confidence = float(confidence.item() if hasattr(confidence, 'item') else confidence)
            
        current_time = float(current_time.item() if hasattr(current_time, 'item') else current_time)
        start_time = max(0, current_time - 3)
        end_time = current_time + 3

        cursor = self.conn.cursor()

        transaction_query = self._load_query("process_incident_segment.sql")
        params = (
            id_user, 
            media_type, 
            start_time, 
            end_time, 
            status, 
            id_source,
            confidence,
            media_path
        )

        try:
            cursor.execute(transaction_query, params)
            result = cursor.fetchone()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"Помилка виконання транзакції інциденту: {e}")
            raise e     
    def get_or_create_file_source(self, file_path):
        filename = os.path.basename(file_path)
        
        cursor = self.conn.cursor()
        try:
            query_get = self._load_query("get_file_source.sql")
            cursor.execute(query_get, (file_path,))
            row = cursor.fetchone()
            
            if row:
                return int(row[0])
            
            print(f"Реєстрація нового файлу: {filename}")
            
            query_create = self._load_query("create_file_source.sql")
            cursor.execute(query_create, (filename, file_path))
            
            row = cursor.fetchone()
            
            if row:
                id_new = int(row[0])
                print(f"Новий файл зареєстровано. ID: {id_new}")
                return id_new
            else:
                print("Помилка: Не вдалося отримати ID нового файлу.")
                return None
        
        except Exception as e:
            print(f"Source Error: {e}")
            return None
    def mark_status(self, id_incident, status='Хибне спрацювання'):
        cursor = self.conn.cursor()
        try:
            query = self._load_query("mark_status.sql")
            cursor.execute(query, (status, id_incident))
            print(f"Інцидент {id_incident} позначено як {status}.")
        except Exception as e:
            print(f"Update Error: {e}")  
    def get_confirmed_incidents(self):
        query = self._load_query("get_confirmed_incidents.sql")

        cursor = self.conn.cursor()
        cursor.execute(query)

        columns = [column[0] for column in cursor.description]
        incidents = []
        
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            
            start_sec = int(data['proof_start']) if data['proof_start'] else 0
            m_start, s_start = divmod(start_sec, 60)
            
            end_sec = int(data['proof_end']) if data['proof_end'] else 0
            m_end, s_end = divmod(end_sec, 60)
            
            data['time_formatted'] = f"{m_start:02d}:{s_start:02d} - {m_end:02d}:{s_end:02d}"
            
            if data['found_time']:
                data['found_time_formatted'] = data['found_time'].strftime("%Y-%m-%d %H:%M:%S")
            else:
                data['found_time_formatted'] = "Невідомо"
                
            incidents.append(data)
            
        return incidents
        