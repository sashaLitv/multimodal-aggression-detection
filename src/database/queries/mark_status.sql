SET NOCOUNT ON;
UPDATE incident SET status = ? 
WHERE id_incident = ?