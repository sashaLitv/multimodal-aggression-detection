CREATE TABLE [dbo].[users] (
    [id_user]      INT            IDENTITY (1, 1) NOT NULL,
    [last_name]    NVARCHAR (255) NOT NULL,
    [first_name]   NVARCHAR (255) NOT NULL,
    [middle_name]  NVARCHAR (255) NOT NULL,
    [profile_type] NVARCHAR (50)  NOT NULL,
    PRIMARY KEY CLUSTERED ([id_user] ASC)
);


GO

