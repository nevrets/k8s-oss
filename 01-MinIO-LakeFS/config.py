class CFG:
    # MinIO credentials and endpoint
    minio_endpoint = 'http://172.7.0.45:30234/'
    access_key = 'minio'
    secret_key = 'minio123'

    # Bucket name
    service_name = 's3'
    bucket_name = 'test20341'
    repository_name = bucket_name
    object_name = 'AT50_e.csv'

    # 파일 경로
    file_path = 'AT50_e.csv'
    download_path = '06-Load-data/download'
    
    lakefs_endpoint = 'http://172.7.0.45:30300/api/v1'
    lakefs_access_key = 'AKIAJVAZ4NFAVXAGXYVQ'
    lakefs_secret_key = '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'
    