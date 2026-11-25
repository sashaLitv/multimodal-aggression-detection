SET NOCOUNT ON;
INSERT INTO audit (id_user, id_source, audit_start, audit_end)
OUTPUT INSERTED.id_audit
VALUES (?, ?, ?, ?);