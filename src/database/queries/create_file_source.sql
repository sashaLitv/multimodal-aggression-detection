SET NOCOUNT ON;
INSERT INTO data_source (name, source_type, connection_path, description)
OUTPUT INSERTED.id_source
VALUES (?, 'File', ?, N'Завантажений користувачем файл');