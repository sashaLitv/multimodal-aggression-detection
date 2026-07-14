import subprocess

import pyodbc
import docker
import time
from docker.errors import NotFound
import os

class DatabaseManager:
    def __init__(self):
        self.server = '127.0.0.1,1433' 
        self.database = 'bullying_monitoring' 
        self.username = 'sa'
        self.password = 'VeryStr0ngP@ssw0rd' 
        self.driver = '{ODBC Driver 17 for SQL Server}'
        self.image_name = "mcr.microsoft.com/azure-sql-edge"
        self.container_name = "sql"

        self.ensure_docker_daemon_running()

        self.ensure_docker_container()
        self.conn = self.get_connection()

    def ensure_docker_daemon_running(self):
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

    def ensure_docker_container(self):
        client = docker.from_env()

        try:
            container = client.containers.get(self.container_name)
            if container.status != "running":
                print(f"Контейнер {self.container_name} знайдено, але він зупинений. Запуск...")
                container.start()
            else:
                print(f"Контейнер {self.container_name} вже працює.")

        except NotFound:
            container = client.containers.run(
                image = self.image_name,
                name = self.container_name,
                detach = True,
                environment = {            
                    'ACCEPT_EULA': 'Y',
                    'MSSQL_SA_PASSWORD': self.password,
                },
                ports={'1433/tcp': 1433}, 
                cap_add = ["SYS_PTRACE"]
            )

        self._wait_for_sql_ready()
    def _wait_for_sql_ready(self):
        max_retries = 15
        conn_str = f'DRIVER={self.driver};SERVER={self.server};UID={self.username};PWD={self.password}'
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
            container = client.containers.get(self.container_name)

            if container.status == "running":
                print(f"Контейнер {self.container_name} знайдено, зупинка..")
                container.stop()
        except NotFound:
            print(f"Контейнер {self.container_name} не знайдено.")
    def get_connection(self):
        connection_string = f'DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}'
        return pyodbc.connect(connection_string, autocommit=True)


    def create_audit_record(self, id_user, id_source, start_time, end_time):
        cursor = self.conn.cursor()
        try:
            query = """
                SET NOCOUNT ON;
                INSERT INTO audit (id_user, id_source, audit_start, audit_end)
                OUTPUT INSERTED.id_audit
                VALUES (?, ?, ?, ?);
            """
            cursor.execute(query, (id_user, id_source, start_time, end_time))
            
            
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

        transaction_query = """
        SET NOCOUNT ON;
        BEGIN TRY
            BEGIN TRAN;

            -- Приймаємо параметри з Python у безпечні SQL-змінні
            DECLARE @p_id_user INT = ?;
            DECLARE @p_media_type NVARCHAR(50) = ?;
            DECLARE @p_start_time BIGINT = ?;
            DECLARE @p_end_time BIGINT = ?;
            DECLARE @p_status NVARCHAR(50) = ?;
            DECLARE @p_id_source INT = ?;
            DECLARE @p_confidence FLOAT = ?;
            DECLARE @p_save_path NVARCHAR(255) = ?; -- ДОДАНО: Шлях до файлу

            DECLARE @existing_id INT;

            -- 1. Правильна математика перетину інтервалів (з вікном ±5 секунд)
            SELECT TOP 1 @existing_id = i.id_incident
            FROM [dbo].[incident] i WITH (UPDLOCK, SERIALIZABLE)
            JOIN [dbo].[media_proof] mp ON i.id_incident = mp.id_incident
            WHERE i.id_user = @p_id_user 
              AND mp.id_source = @p_id_source
              AND (mp.proof_start <= @p_end_time + 5 AND mp.proof_end >= @p_start_time - 5);

            IF @existing_id IS NOT NULL
            BEGIN
                -- Оновлюємо статус та перезаписуємо впевненість (беремо максимальну)
                UPDATE [dbo].[incident]
                SET status = @p_status,
                    confidence = CASE 
                                    WHEN @p_confidence > confidence THEN @p_confidence 
                                    ELSE confidence 
                                 END
                WHERE id_incident = @existing_id;

                DECLARE @existing_proof_index INT;
                
                -- 2. Шукаємо, чи є доказ з ТОГО Ж САМОГО файлу поруч
                SELECT TOP 1 @existing_proof_index = media_index
                FROM [dbo].[media_proof]
                WHERE id_incident = @existing_id 
                  AND id_source = @p_id_source
                  AND (proof_start <= @p_end_time + 5 AND proof_end >= @p_start_time - 5);

                IF @existing_proof_index IS NOT NULL
                BEGIN
                    -- 3А. Доказ є! Просто розширюємо його межі
                    UPDATE [dbo].[media_proof]
                    SET proof_start = CASE WHEN @p_start_time < proof_start THEN @p_start_time ELSE proof_start END,
                        proof_end   = CASE WHEN @p_end_time > proof_end THEN @p_end_time ELSE proof_end END
                    WHERE id_incident = @existing_id 
                      AND id_source = @p_id_source 
                      AND media_index = @existing_proof_index;
                END
                ELSE
                BEGIN
                    -- 3Б. Додаємо новий доказ до ЦЬОГО Ж інциденту.
                    DECLARE @next_index INT;
                    SELECT @next_index = ISNULL(MAX(media_index), 0) + 1 
                    FROM [dbo].[media_proof] 
                    WHERE id_incident = @existing_id AND id_source = @p_id_source;

                    -- ВИПРАВЛЕНО: Додано media_type та save_path
                    INSERT INTO [dbo].[media_proof] (id_incident, id_source, media_index, media_type, save_path, proof_start, proof_end)
                    VALUES (@existing_id, @p_id_source, @next_index, @p_media_type, @p_save_path, @p_start_time, @p_end_time);
                END

                SELECT @existing_id AS result_id;
            END
            ELSE
            BEGIN
                -- 4. Якщо взагалі нічого немає в ці ±5 секунд - створюємо новий інцидент
                DECLARE @new_id_table TABLE (id INT);

                INSERT INTO [dbo].[incident] (id_user, status, confidence, fixation_time)
                OUTPUT INSERTED.id_incident INTO @new_id_table
                VALUES (@p_id_user, @p_status, @p_confidence, GETDATE());

                DECLARE @new_id INT = (SELECT TOP 1 id FROM @new_id_table);

                -- ВИПРАВЛЕНО: Додано save_path
                INSERT INTO [dbo].[media_proof] (id_incident, id_source, media_type, media_index, save_path, proof_start, proof_end)
                VALUES (@new_id, @p_id_source, @p_media_type, 1, @p_save_path, @p_start_time, @p_end_time);

                SELECT @new_id AS result_id;

            END

            COMMIT TRAN;
        END TRY
        BEGIN CATCH
            IF @@TRANCOUNT > 0
                ROLLBACK TRAN;
            THROW;
        END CATCH
        """

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
            self.conn.commit()
            
            return result[0] if result else None
            
        except Exception as e:
            self.conn.rollback()
            print(f"Помилка виконання транзакції інциденту: {e}")
            raise e
        
    def get_or_create_file_source(self, file_path):
        filename = os.path.basename(file_path)
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("SET NOCOUNT ON; SELECT id_source FROM data_source WHERE connection_path = ?", (file_path,))
            row = cursor.fetchone()
            
            if row:
                return int(row[0])
            
            print(f"Реєстрація нового файлу: {filename}")
            
            query = """
                SET NOCOUNT ON;
                INSERT INTO data_source (name, source_type, connection_path, description)
                OUTPUT INSERTED.id_source
                VALUES (?, 'File', ?, 'Завантажений користувачем файл');
            """
            cursor.execute(query, (filename, file_path))
            
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
            query = """
                SET NOCOUNT ON;
                UPDATE incident SET status = ? WHERE id_incident = ?
            """
            cursor.execute(query, (status, id_incident))
            self.conn.commit()
            print(f"Інцидент {id_incident} позначено як {status}.")
        except Exception as e:
            self.conn.rollback()
            print(f"Update Error: {e}")
    
    def get_confirmed_incidents(self):
        query = """
            SELECT 
                i.id_incident,
                mp.media_type as media_type,
                (u.last_name + ' ' + u.first_name) as user_name,
                ds.connection_path as source_name,
                mp.proof_start,
                mp.proof_end,
                i.fixation_time as found_time,
                i.confidence
            FROM 
                [dbo].[incident] i
            JOIN 
                [dbo].[users] u ON i.id_user = u.id_user
            JOIN 
                [dbo].[media_proof] mp ON i.id_incident = mp.id_incident
            JOIN 
                [dbo].[data_source] ds ON mp.id_source = ds.id_source
            WHERE 
                i.status = N'Підтверджено' OR i.status = N'Критично'
            ORDER BY 
                i.id_incident DESC
        """

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
        