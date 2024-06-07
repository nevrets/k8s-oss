import os
import base64
import boto3

# configuration python file
from config import CFG

class MinIO:
    def __init__(self):
        self.minio_bucket_name = CFG.minio_bucket_name
        self.minio_object_name = CFG.minio_object_name
        self.object_path = CFG.object_path
        
        self.minio_client = boto3.client(
            CFG.minio_service_name, 
            endpoint_url=CFG.minio_endpoint, 
            aws_access_key_id=CFG.minio_access_key, 
            aws_secret_access_key=CFG.minio_secret_key
        )
        
    ''' CREATE BUCKET '''
    def create_bucket(self):
        self.minio_client.create_bucket(Bucket=self.minio_bucket_name)
        print(f'bucket name: {self.minio_bucket_name}')


    ''' UPLOAD OBJECT - 1 '''
    def upload_object(self):
        with open(self.object_path, 'rb') as f:
            object = f.read()
    
        minio_key = f'{self.minio_bucket_name}/{self.minio_object_name}'
        self.minio_client.put_object(Bucket=self.minio_bucket_name,
                                     Key=minio_key,
                                     Body=object)
        
        # 객체 목록 가져오기
        response = self.minio_client.list_objects(Bucket=self.minio_bucket_name)

        # 객체 목록 출력
        print("Objects in bucket:")
        for obj in response.get('Contents', []):
            print(obj['Key'])
    
    
    
    ''' UPLOAD OBJECT - 2 '''
    def upload_fileobj(self):
        try:
            with open(self.object_path, 'rb') as f:
                self.minio_client.upload_fileobj(f, self.minio_bucket_name, 'AT50_e.csv')
                print(f'File uploaded to {self.minio_bucket_name}/{self.minio_object_name}')
                
        except FileNotFoundError:
            print(f'File not found: {self.object_path}')
            return
        
        # 객체 목록 가져오기
        response = self.minio_client.list_objects(Bucket=self.minio_bucket_name)

        # 객체 목록 출력
        print("Objects in bucket:")
        for obj in response.get('Contents', []):
            print(obj['Key'])

        
        
    ''' DOWNLOAD OBJECT '''
    def download_object(self):
        self.minio_client.download_file(CFG.minio_bucket_name,
                                        CFG.minio_object_name, 
                                        CFG.data_path)

    ''' GET OBJECT '''
    def get_content_in_object(self):
        response = self.minio_client.get_object(Bucket=CFG.minio_bucket_name, 
                                                Key=CFG.minio_object_name)
        
        data = response['Body'].read().decode('utf-8')

        return data



if __name__ == '__main__':
    minio = MinIO()
    
    minio.create_bucket()
    minio.upload_object()
    # minio.upload_fileobj()
    print('')