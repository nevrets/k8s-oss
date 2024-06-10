import base64
import json
import requests

import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient

from py2minio import MinIO
from config import CFG


class LakeFS:
    def __init__(self):
        self.lakefs_endpoint = CFG.lakefs_endpoint
        self.lakefs_access_key = CFG.lakefs_access_key    # ID
        self.lakefs_secret_key = CFG.lakefs_secret_key    # PW
        
        self.lakefs_repository_name = CFG.lakefs_repository_name
        self.lakefs_branch_name = CFG.lakefs_branch_name
        
        self.minio_service_name = CFG.minio_service_name
        self.auth_header = self.get_auth_header()

        # lakeFS credentials and endpoint
        self.configuration = lakefs_client.Configuration()
        self.configuration.username = CFG.lakefs_access_key
        self.configuration.password = CFG.lakefs_secret_key
        self.configuration.host = CFG.lakefs_url


    ''' CREATE REPOSITORY '''
    def create_repository(self):
        # minio = connect_minio()
        client = LakeFSClient(self.configuration)

        try:
        # Create repository
            repository_creation = models.RepositoryCreation(name=self.lakefs_repository_name, 
                                                            storage_namespace=f'{self.minio_service_name}://{self.lakefs_repository_name}', 
                                                            default_branch=self.lakefs_branch_name)
            
            client.repositories.create_repository(repository_creation)
            print("Repository created successfully.")

        except lakefs_client.ApiException as e:
            print("Exception when calling RepositoriesApi -> create_repository: %s\n" % e)


    ''' UPLOAD FILE '''
    def upload_file(self, file_path):
        client = LakeFSClient(self.configuration)
        
        with open(file_path, 'rb') as f:
            client.objects.upload_object(repository=self.lakefs_repository_name, 
                                         branch=self.lakefs_branch_name, 
                                         path='data', content=f)


    ''' DOWNLOAD FILE '''
    def get_lakefs_file_content(self, file_path):
        client = LakeFSClient(self.configuration)
        
        try:
            # lakeFS에서 파일 가져오기
            response = client.objects.get_object(repository=self.lakefs_repository_name, 
                                                 ref=self.lakefs_branch_name, 
                                                 path=file_path)
            
            # 파일 내용 읽기
            file_content = response.read().decode('utf-8')
            
            return file_content

        except Exception as e:
            print("Error:", e)
            return None


    ''' COMMIT LAKEFS '''
    def commit_to_lakefs(self, commit_message):
        commit_url = f'{self.lakefs_endpoint}/repositories/{self.lakefs_repository_name}/branches/{self.lakefs_branch_name}/commits'
        commit_payload = {'message': f'{commit_message}'}

        commit_headers = {
            'Authorization': f'Basic {self.auth_header}',
            'Content-Type': 'application/json'
        }

        commit_response = requests.post(commit_url, headers=commit_headers, data=json.dumps(commit_payload))

        if commit_response.status_code == 201:
            msg = 'Commit successful'
        else:
            msg = f'Commit failed: {commit_response.text}'
            
        return msg


    def get_auth_header(self):
        auth_string = f'{self.lakefs_access_key}:{self.lakefs_secret_key}'
        return base64.b64encode(auth_string.encode()).decode()



if __name__ == '__main__':
    lakefs = LakeFS()
    minio = MinIO()
    
    minio.create_bucket()
    lakefs.create_repository()
    
    lakefs.upload_file(CFG.object_path)
    
    file_content = lakefs.get_lakefs_file_content('data')
    
    msg = lakefs.commit_to_lakefs('20240610 test')
    
    print('')