import requests
import json
import re

files_endpt = "https://api.gdc.cancer.gov/files"
filters = {
    "op": "and",
    "content":[
        {
            "op":"in",
            "content":{
                "field":"files.data_category",
                "value":"DNA Methylation"
            }
        },
        {
            "op": "in",
            "content":{
                "field": "cases.project.project_id",
                "value": "TCGA-BRCA"
                }
        },
                {
            "op": "in",
            "content":{
                "field": "files.platform",
                "value": "illumina human methylation 27"
                }
        }
    ]
}
params = {
    "filters": json.dumps(filters),
    "fields": "file_id",
    "format": "JSON",
    "size": "1000"
    }
response = requests.get(files_endpt, params = params)

file_uuid_list = []

# This step populates the download list with the file_ids from the previous query
for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
    file_uuid_list.append(file_entry["file_id"])

file_uuid_list = file_uuid_list[:10]  

data_endpt = "https://api.gdc.cancer.gov/data"

params = {"ids": file_uuid_list}

print('here')
print(params)
print(file_uuid_list)
response = requests.post(data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"})

response_head_cd = response.headers["Content-Disposition"]

file_name = re.findall("filename=(.+)", response_head_cd)[0]

with open(file_name, "wb") as output_file:
    output_file.write(response.content)