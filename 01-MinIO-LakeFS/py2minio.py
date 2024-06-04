import os
import boto3

# configuration python file
import config


''' CONNECT MINIO '''
def connect_minio():
    # Create a MinIO client
    minio_client = boto3.client(config.CFG.ervice_name, 
                                endpoint_url=config.CFG.minio_endpoint, 
                                aws_access_key_id=config.CFG.access_key, 
                                aws_secret_access_key=config.CFG.secret_key)
    
    return minio_client


''' CREATE BUCKET '''
def create_bucket(minio_client):
    minio_client.create_bucket(Bucket=config.CFG.bucket_name)
    print(f'bucket name: {config.CFG.bucket_name}')


''' UPLOAD OBJECT'''
def upload_object(minio_client):
    '''
    with open(config.CFG.file_path, 'rb') as f:
        file_content = f.read()
    
    minio_client.put_object(Bucket=config.CFG.bucket_name, 
                            Key=config.CFG.minio_key, 
                            Body=file_content)
    
    '''
    
    try:
        with open(config.CFG.file_path, 'rb') as f:
            minio_client.upload_fileobj(f, config.CFG.bucket_name, config.CFG.file_path)
        print(f'File uploaded to {config.CFG.bucket_name}/{config.CFG.minio_key}')
            
    except FileNotFoundError:
        print(f'File not found: {config.CFG.file_path}')
        return
    
    # 객체 목록 가져오기
    response = minio_client.list_objects(Bucket=config.CFG.bucket_name)

    # 객체 목록 출력
    print("Objects in bucket:")
    for obj in response.get('Contents', []):
        print(obj['Key'])


''' DOWNLOAD OBJECT '''
def download_object(minio_client):
    # download_file_path = os.path.join(config.CFG.download_path, os.path.basename(config.CFG.file_path))
    
    minio_client.download_file(config.CFG.bucket_name,
                               config.CFG.object_name, 
                               config.CFG.download_path)

''' GET OBJECT '''
def get_content_in_object(minio_client):
    response = minio_client.get_object(Bucket=config.CFG.bucket_name, 
                                       Key=config.CFG.object_name)
    
    data = response['Body'].read().decode('utf-8')

    return data



if __name__ == '__main__':
    minio_client = connect_minio()
    
    create_bucket(minio_client)
    upload_object(minio_client)
    download_object(minio_client)
    
    data = get_content_in_object(minio_client)
    print("Object Content:")
    print(data)