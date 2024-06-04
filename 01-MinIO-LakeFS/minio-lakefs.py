import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient

import boto3


# MinIO credentials and endpoint
minio_endpoint = 'http://172.7.0.45:30234/'
access_key = 'minio'
secret_key = 'minio123'

# Bucket name
bucket_name = 'test1234567'

# 파일 경로
file_path = 'LP=39.001331,129.548859 IP=36.738583,128.414001 Time=349.csv'

# 수정 필요
remain = True


# lakeFS credentials and endpoint
configuration = lakefs_client.Configuration()
configuration.username = 'AKIAJVAZ4NFAVXAGXYVQ'
configuration.password = '6ziXTIJbFa9fwRllPF6Vpz9E0Y7HjfOEJcTkVKt+'
configuration.host = 'http://172.7.0.45:30300'




def connect_minio():
    # Create a MinIO client
    minio_client = boto3.client('s3', endpoint_url=minio_endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    # Create bucket
    # if not remain:
    minio_client.create_bucket(Bucket=bucket_name)

    # 파일 업로드
    with open(file_path, 'rb') as f:
        minio_client.upload_fileobj(f, bucket_name, file_path)

    # 객체 목록 가져오기
    response = minio_client.list_objects(Bucket=bucket_name)

    # 객체 목록 출력
    print("Objects in bucket:")
    for obj in response.get('Contents', []):
        print(obj['Key'])

    return response

## minio 연결해서 데이터 가져오는 법


def connect_lakefs():
    # minio = connect_minio()
    client = LakeFSClient(configuration)

    # lakeFS 클라이언트 객체 생성
    # client = lakefs_client.ApiClient(configuration)

    # RepositoryApi 객체 생성
    # repositories_api = lakefs_client.api.repositories_api.RepositoriesApi(client)
    # repo = models.RepositoryCreation(name='example-repo', storage_namespace='s3://storage-bucket/repos/example-repo', default_branch='main')

    try:
        # Create repository
        repository_creation = models.RepositoryCreation(name='test1234567', storage_namespace='s3://test1234567', default_branch='main')
        client.repositories.create_repository(repository_creation)

        print("Repository created successfully.")

    except lakefs_client.ApiException as e:
        print("Exception when calling RepositoriesApi->create_repository: %s\n" % e)


    # 1. 단순 파일 업로드
    # # with open('file.csv', 'rb') as f:
    #     client.objects.upload_object(repository='example-repo', branch='experiment-aggregations1', path='path/to/file.csv', content=f)



    print()


# dummy 확인
if __name__ == '__main__':
    # config = get_config()
    connect_lakefs()