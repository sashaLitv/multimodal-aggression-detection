CREATE TABLE [dbo].[audit] (
    [id_audit]    INT      IDENTITY (1, 1) NOT NULL,
    [id_user]     INT      NOT NULL,
    [id_source]   INT      NOT NULL,
    [audit_start] DATETIME NOT NULL,
    [audit_end]   DATETIME NULL,
    PRIMARY KEY CLUSTERED ([id_audit] ASC),
    CONSTRAINT [FK_audit_source] FOREIGN KEY ([id_source]) REFERENCES [dbo].[data_source] ([id_source]),
    CONSTRAINT [FK_audit_user] FOREIGN KEY ([id_user]) REFERENCES [dbo].[users] ([id_user])
);


GO

