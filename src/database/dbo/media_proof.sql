CREATE TABLE [dbo].[media_proof] (
    [id_incident]          INT            NOT NULL,
    [id_source]            INT            NOT NULL,
    [media_index]          INT            NOT NULL,
    [media_type]           NVARCHAR (15)  NOT NULL,
    [save_path]            NVARCHAR (255) NOT NULL,
    [proof_start]          BIGINT         NOT NULL,
    [proof_end]            BIGINT         NOT NULL,
    [fixation_description] NVARCHAR (MAX) NULL,
    CONSTRAINT [PK_media_proof] PRIMARY KEY CLUSTERED ([id_incident] ASC, [id_source] ASC, [media_index] ASC),
    CONSTRAINT [FK_media_incident] FOREIGN KEY ([id_incident]) REFERENCES [dbo].[incident] ([id_incident]) ON DELETE CASCADE,
    CONSTRAINT [FK_media_source] FOREIGN KEY ([id_source]) REFERENCES [dbo].[data_source] ([id_source])
);


GO

