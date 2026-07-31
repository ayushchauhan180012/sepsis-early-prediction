from fastapi import FastAPI, HTTPException , Path , Query
from pydantic import BaseModel, computed_field, Field
from typing import Annotated, Optional
from fastapi.responses import JSONResponse
import json
app=FastAPI()
# pyd model for newly created data
class Patient(BaseModel):
# inside field "..." means required: input must not be empty
    id: Annotated[str, Field(...,description='ID of patient',example="P001")]
    name: Annotated[str, Field(..., description='name of patient')]
    age: Annotated[int, Field(...,gt=0,lt=120,description='age of the patient')]
    height: Annotated[float, Field(...,gt=0,description='height of the patient in metres')]
    weight: Annotated[float, Field(...,gt=0,description='weight of the patient in kgs')]
    
    @computed_field
    @property
    def bmi(self) -> float:
        bmi=round(self.weight/(self.height**2),2)
        return bmi

class PatientUpdate(BaseModel):

    name: Annotated[Optional[str], Field(default=None, description='name of patient')]
    age: Annotated[Optional[int], Field(default=None,gt=0,lt=120,description='age of the patient')]
    height: Annotated[Optional[float], Field(default=None,gt=0,description='height of the patient in metres')]
    weight: Annotated[Optional[float], Field(default=None,gt=0,description='weight of the patient in kgs')]
    

def load_data():
    with open('data.json','r') as f:
        data=json.load(f)
    return data
def save_data(data):
    with open('data.json','w') as f:
        json.dump(data,f)


@app.get("/")
def hello():
    return {'message':'Patient management system API'}

@app.get("/about")
def about():
    return {'message':'Managing Patient recrds'}

@app.get('/view')
def view():
    data=load_data()

    return data

@app.get('/patient/{patient_id}')
def view_patient(patient_id: str = Path(..., description='Id of patient' ,example='p001')):
    # load all patient
    data = load_data()

    if patient_id in data:
        return data[patient_id]
    raise HTTPException(status_code=404 , detail= 'patient not found')
    
@app.get('/sort')
def sort_patients(sort_by: str = Query(..., description='sort on the basis of height, weight or bmi') , order: str=Query('asc', description='sort in ascending order')):
    
    valid_fields= ['height', 'weight' ,'bmi']
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400 ,detail='invalid fields select from {valid_fields}')
    
    if order not in ['asc', 'dec']:
        raise HTTPException(status_code=400, detail='invalid order select between asc or dec')
    
    data = load_data()

    sort_order= True if order=='dec' else False

    sorted_data= sorted(data.values(), key=lambda x: x.get(sort_by, 0), reverse=sort_order)

    return sorted_data

@app.post('/create')

def create_pt(patient: Patient):
    # load 1st
    data= load_data()

    # check if patient already exists
    if patient.id in data:
        raise HTTPException(status_code=400, detail='patient already exists')
    
    # new patient adds to database
    data[patient.id]=patient.model_dump()

    # save into json
    save_data(data)

    return JSONResponse(status_code=200, content={'message':'patient created successfully'})


@app.put("/edit/{patient_id}")
def update_patient(patient_id: str, patient_update: PatientUpdate):

    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail="Patient not found")

    existing_patient_info = data[patient_id]

    update_info = patient_update.model_dump(exclude_unset=True)

    for key, value in update_info.items():
        existing_patient_info[key] = value

    existing_patient_info["id"] = patient_id

    updated_patient = Patient(**existing_patient_info)

    data[patient_id] = updated_patient.model_dump()

    save_data(data)

    return JSONResponse(
        status_code=200,
        content={"message": "Patient updated successfully"}
    )

@app.delete('/delete/{patient_id}')
def delete_patient(patient_id: str):
    
    data=load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail="patient not found")
    
    del data[patient_id]

    save_data(data)

    return JSONResponse(status_code=200, content="successfully deleted record")

