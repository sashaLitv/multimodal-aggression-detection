CREATE TABLE [dbo].[incident] (
    [id_incident]   INT           IDENTITY (1, 1) NOT NULL,
    [fixation_time] DATETIME2 (7) NOT NULL,
    [status]        NVARCHAR (50) NOT NULL,
    [confidence]    FLOAT (53)    NOT NULL,
    [id_user]       INT           NOT NULL,
    PRIMARY KEY CLUSTERED ([id_incident] ASC),
    CONSTRAINT [CK_status] CHECK ([status]=N'Вирішено' OR [status]=N'Критично' OR [status]=N'Хибна тривога' OR [status]=N'Підтверджено'),
    CONSTRAINT [FK_incident_user] FOREIGN KEY ([id_user]) REFERENCES [dbo].[users] ([id_user])
);


GO

