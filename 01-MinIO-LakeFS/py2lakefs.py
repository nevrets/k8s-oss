import hashlib
import base64
import json
import boto3
import requests
# from lakefs_client import Configuration, LakeFSClient

import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient

from config import CFG

# lakeFS credentials and endpoint
configuration = lakefs_client.Configuration()
configuration.username = 'AKIAJVAZ4NFAVXAGXYVQ'
configuration.password = '6ziXTIJbFa9fwRllPF6Vpz9E0Y7HjfOEJcTkVKt+'
configuration.host = 'http://172.7.0.45:30300'


class LakeFS:
    def __init__(self):
        self.lakefs_endpoint = CFG.lakefs_endpoint
        self.lakefs_access_key = CFG.lakefs_access_key    # ID
        self.lakefs_secret_key = CFG.lakefs_secret_key    # PW
        
        self.repository = CFG.lakefs_repository_name
        self.bucketname = CFG.lakefs_bucket_name
        self.branch = CFG.branch
    
        self.auth_header = self.get_auth_header()
        
        self.minio_client = boto3.client(
            CFG.service_name,
            endpoint_url=CFG.minio_endpoint,
            aws_access_key_id=CFG.minio_access_key,
            aws_secret_access_key=CFG.minio_secret_key
        )

    def connect_lakefs():
        # minio = connect_minio()
        client = LakeFSClient(configuration)

        return client

    def create_repository(client):
        try:
            # Create repository
            repository_creation = models.RepositoryCreation(name=CFG.repository_name, 
                                                            storage_namespace=f's3://{CFG.bucket_name}', 
                                                            default_branch='main')
            
            client.repositories.create_repository(repository_creation)

            print("Repository created successfully.")

        except lakefs_client.ApiException as e:
            print("Exception when calling RepositoriesApi->create_repository: %s\n" % e)

    def get_auth_header(access_key, secret_key):
        auth_string = f'{access_key}:{secret_key}'
        return base64.b64encode(auth_string.encode()).decode()

    def check_and_create_branch(lakefs_endpoint, auth_header, repository, branch):
        branch_url = f'{lakefs_endpoint}/repositories/{repository}/branches/{branch}'
        branch_headers = {'Authorization': f'Basic {auth_header}'}

        branch_response = requests.get(branch_url, headers=branch_headers)

        if branch_response.status_code == 404:
            create_branch_url = f'{lakefs_endpoint}/repositories/{repository}/branches'
            create_branch_payload = {'name': branch, 'source': 'main'}

            create_branch_response = requests.post(create_branch_url, headers={
                'Authorization': f'Basic {auth_header}',
                'Content-Type': 'application/json'
            }, data=json.dumps(create_branch_payload))

            if create_branch_response.status_code == 201:
                print(f'Branch {branch} created successfully')
            else:
                print(f'Failed to create branch: {create_branch_response.text}')
                return False
            
        elif branch_response.status_code == 200:
            print(f'Branch {branch} exists')
        
        else:
            print(f'Error checking branch: {branch_response.text}')
            return False
        
        return True


    def upload_file_to_minio(minio_client, 
                            bucket_name, 
                            minio_key, 
                            file_content):
        
        minio_client.put_object(Bucket=bucket_name, 
                                Key=minio_key, 
                                Body=file_content)
        
        print(f'File uploaded to {bucket_name}/{minio_key}')


    def register_file_in_lakefs(lakefs_endpoint, 
                                auth_header, 
                                repository, 
                                branch, 
                                path, 
                                physical_address, 
                                file_content):
        
        upload_url = f'{lakefs_endpoint}/repositories/{repository}/branches/{branch}/objects'
        upload_headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/json'
        }

        md5_checksum = hashlib.md5(file_content).hexdigest()
        upload_data = {
            'physical_address': physical_address,
            'checksum': md5_checksum,
            'size_bytes': len(file_content),
            'content_type': 'application/octet-stream'
        }

        upload_response = requests.put(upload_url, 
                                    headers=upload_headers, 
                                    params={'path': path}, 
                                    data=json.dumps(upload_data))

        if upload_response.status_code == 201:
            print('File metadata registered successfully in lakeFS')
            
        else:
            print(f'File metadata registration failed: {upload_response.text}')
            return False
        
        return True


    def commit_to_lakefs(lakefs_endpoint, 
                        auth_header, 
                        repository, 
                        branch):
        commit_url = f'{lakefs_endpoint}/repositories/{repository}/branches/{branch}/commits'
        commit_payload = {
            'message': 'Initial commit'
        }

        commit_headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/json'
        }

        commit_response = requests.post(commit_url, 
                                        headers=commit_headers, 
                                        data=json.dumps(commit_payload))

        if commit_response.status_code == 201:
            print('Commit successful')
            
        else:
            print(f'Commit failed: {commit_response.text}')
            return False
        
        return True





def main():
    lakefs_endpoint = 'http://172.7.0.45:30300/api/v1'
    lakefs_access_key = 'AKIAJVAZ4NFAVXAGXYVQ'
    lakefs_secret_key = '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'
    
    repository = 'tmp1'
    branch = '20240517-2'
    bucket_name = 'test20341'
    file_path = 'AT50_e.csv'
    path = file_path

    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return

    auth_header = get_auth_header(lakefs_access_key, lakefs_secret_key)

    minio_client = boto3.client(
        's3',
        endpoint_url='http://172.7.0.45:30234',
        aws_access_key_id="minio",
        aws_secret_access_key="minio123"
    )

    if not check_and_create_branch(lakefs_endpoint, auth_header, repository, branch):
        return 

    s3_key = f'{repository}/{branch}/{path}'
    upload_file_to_minio(minio_client, 
                         bucket_name, 
                         s3_key, 
                         file_content)

    physical_address = f's3://{bucket_name}/{s3_key}'

    if not register_file_in_lakefs(lakefs_endpoint, auth_header, repository, branch, path, physical_address, file_content):
        return 

    if not commit_to_lakefs(lakefs_endpoint, auth_header, repository, branch):
        return 


if __name__ == '__main__':
    main()
    print("Success")
