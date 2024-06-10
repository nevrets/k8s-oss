import pandas as pd
import requests
import json
import pytz
import uuid

from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers

from py2minio import MinIO
from config import CFG


class Elastic:
    def __init__(self, index_name):
        self.repository_name = CFG.lakefs_repository_name
        self.branch_name = CFG.lakefs_branch_name
        
        self.elasticsearch = Elasticsearch(CFG.elasticsearch_url)
        self.index_name = index_name
        self.interval = 1    # 간격 설정(1초)
        
        self.kibana_url = CFG.kibana_url
        

    def create_elasticsearch_index(self, df):
        properties = {
            "Time": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}
        }
        
        for col in df.columns:
            if col != "Time":
                if df[col].dtype == 'float64':
                    properties[col] = {"type": "float"}
                elif df[col].dtype == 'int64':
                    properties[col] = {"type": "long"}
                else:
                    properties[col] = {"type": "text"}
        
        if self.elasticsearch.indices.exists(index=self.index_name):
            index_mapping = {
                "properties": properties
            }
            
            # 데이터프레임의 각 열에 대한 매핑 생성
            self.elasticsearch.indices.put_mapping(index=self.index_name, body=index_mapping)
            print(f"Index '{self.index_name}' created successfully.")
            
        else:
            index_mapping = {
                "mappings": {
                    "properties": properties
                }
            }
            
            self.elasticsearch.indices.create(index=self.index_name, body=index_mapping)
            
            
    def index_data_to_es(self, df):
        # KST 시간대로 변환하여 "Time" 열의 값을 datetime 형식으로 변환
        kst = pytz.timezone('Asia/Seoul')
        start_time = datetime.now(tz=kst)
        
        time_intervals = [start_time + timedelta(seconds=i * self.interval) for i in range(len(df))]
        df['Time'] = time_intervals

        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "Time": row['Time'].strftime("%Y-%m-%d %H:%M:%S"),
                    **{col: row[col] for col in df.columns if col != 'Time'}
                }
            }
            for _, row in df.iterrows()
        ]
        
        # 여러 명령을 배치로 수행하기 위한 bulk api
        helpers.bulk(self.elasticsearch, actions)
        print("Data indexed successfully.")

        # 데이터프레임의 마지막 행의 시간을 종료 시간으로 설정
        end_time = df.iloc[-1]['Time']

        # 시작 시간과 종료 시간을 Unix 타임스탬프로 변환하여 반환
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        print(start_timestamp, end_timestamp)
        
        return start_timestamp, end_timestamp
    
    # kibana에서 인덱스 패턴 생성
    def create_kibana_index_pattern(self):
        kibana_payload = {
            "attributes": {
                "title": self.index_name,
                "timeFieldName": "Time"
            }
        }

        kibana_headers = {
            "Content-Type": "application/json",
            "kbn-xsrf": "true"  # Kibana의 CSRF 보호를 우회하기 위해 필요
        }

        response = requests.post(
            f"{self.kibana_url}/api/saved_objects/index-pattern",
            headers=kibana_headers,
            data=json.dumps(kibana_payload)
        )

        if response.status_code == 200:
            print("Kibana index pattern created successfully.")
        else:
            print(f"Failed to create Kibana index pattern: {response.content}")



if __name__ == '__main__':
    es = Elastic('test')
    augment_uuid = str(uuid.uuid4())
    df = pd.read_csv(CFG.object_path)
    df['augment_uuid'] = augment_uuid
    
    print('')