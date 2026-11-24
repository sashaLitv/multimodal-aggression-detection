CREATE TABLE [dbo].[data_source] (
    [id_source]       INT            IDENTITY (1, 1) NOT NULL,
    [name]            NVARCHAR (100) NOT NULL,
    [source_type]     NVARCHAR (15)  NOT NULL,
    [connection_path] NVARCHAR (255) NOT NULL,
    [description]     NVARCHAR (500) NULL,
    PRIMARY KEY CLUSTERED ([id_source] ASC)
);


GO

