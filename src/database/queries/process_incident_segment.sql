SET NOCOUNT ON;
BEGIN TRY
    BEGIN TRAN;

    DECLARE @p_id_user INT = ?;
    DECLARE @p_media_type NVARCHAR(50) = ?;
    DECLARE @p_start_time BIGINT = ?;
    DECLARE @p_end_time BIGINT = ?;
    DECLARE @p_status NVARCHAR(50) = ?;
    DECLARE @p_id_source INT = ?;
    DECLARE @p_confidence FLOAT = ?;
    DECLARE @p_save_path NVARCHAR(255) = ?;

    DECLARE @existing_id INT;

    SELECT TOP 1 @existing_id = i.id_incident
    FROM [dbo].[incident] i WITH (UPDLOCK, SERIALIZABLE)
    JOIN [dbo].[media_proof] mp ON i.id_incident = mp.id_incident
    WHERE i.id_user = @p_id_user
      AND mp.id_source = @p_id_source
      AND (mp.proof_start <= @p_end_time + 5 AND mp.proof_end >= @p_start_time - 5);

    IF @existing_id IS NOT NULL
    BEGIN
        UPDATE [dbo].[incident]
        SET status = @p_status,
            confidence = CASE
                            WHEN @p_confidence > confidence THEN @p_confidence
                            ELSE confidence
                         END
        WHERE id_incident = @existing_id;

        DECLARE @existing_proof_index INT;

        SELECT TOP 1 @existing_proof_index = media_index
        FROM [dbo].[media_proof]
        WHERE id_incident = @existing_id
          AND id_source = @p_id_source
          AND (proof_start <= @p_end_time + 5 AND proof_end >= @p_start_time - 5);

        IF @existing_proof_index IS NOT NULL
        BEGIN
            UPDATE [dbo].[media_proof]
            SET proof_start = CASE WHEN @p_start_time < proof_start THEN @p_start_time ELSE proof_start END,
                proof_end = CASE WHEN @p_end_time > proof_end THEN @p_end_time ELSE proof_end END
            WHERE id_incident = @existing_id
              AND id_source = @p_id_source
              AND media_index = @existing_proof_index;
        END
        ELSE
        BEGIN
            DECLARE @next_index INT;
            SELECT @next_index = ISNULL(MAX(media_index), 0) + 1
            FROM [dbo].[media_proof]
            WHERE id_incident = @existing_id AND id_source = @p_id_source;

            INSERT INTO [dbo].[media_proof] (id_incident, id_source, media_index, media_type, save_path, proof_start, proof_end)
            VALUES (@existing_id, @p_id_source, @next_index, @p_media_type, @p_save_path, @p_start_time, @p_end_time);
        END

        SELECT @existing_id AS result_id;
    END
    ELSE
    BEGIN
        DECLARE @new_id_table TABLE (id INT);

        INSERT INTO [dbo].[incident] (id_user, status, confidence, fixation_time)
        OUTPUT INSERTED.id_incident INTO @new_id_table
        VALUES (@p_id_user, @p_status, @p_confidence, GETDATE());

        DECLARE @new_id INT = (SELECT TOP 1 id FROM @new_id_table);

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