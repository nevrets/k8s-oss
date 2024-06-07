class CFG:
    # MinIO credentials and endpoint
    minio_endpoint = 'http://172.7.0.45:30234/'
    minio_access_key = 'minio'
    minio_secret_key = 'minio123'

    # Bucket name
    minio_service_name = 's3'
    minio_bucket_name = 'nevret'
    minio_object_name = 'AT50_e.csv'

    # LakeFS
    lakefs_endpoint = 'http://172.7.0.45:30300/api/v1'
    lakefs_access_key = 'AKIAJVAZ4NFAVXAGXYVQ'
    lakefs_secret_key = '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'
    
    lakefs_branch_name = '20240607'
    lakefs_repository_name = minio_bucket_name
    
    
    # 파일 경로
    object_path = './01-MinIO-LakeFS/data/AT50_e.csv'
    