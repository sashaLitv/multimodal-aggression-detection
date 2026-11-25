SELECT
    i.id_incident,
    mp.media_type,
    (u.last_name + N' ' + u.first_name) AS user_name,
    ds.connection_path AS source_name,
    mp.proof_start,
    mp.proof_end,
    i.fixation_time AS found_time,
    i.confidence
FROM [dbo].[incident] i
JOIN [dbo].[users] u ON i.id_user = u.id_user
JOIN [dbo].[media_proof] mp ON i.id_incident = mp.id_incident
JOIN [dbo].[data_source] ds ON mp.id_source = ds.id_source
WHERE i.status IN (N'Підтверджено', N'Критично')
ORDER BY i.id_incident DESC;